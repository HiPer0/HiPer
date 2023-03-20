import argparse
import os
import torch
from torch import autocast
from diffusers import DDIMScheduler
from diffusers_ import StableDiffusionPipeline
from accelerate.utils import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="text-inversion-model",
        help="The directory where the target embeddings are saved.",
    )
    parser.add_argument(
        "--personalized_emb_dir",
        type=str,
        default="text-inversion-model",
        help="The directory where the source embeddings are saved.",
    )
    parser.add_argument(
        "--target_txt",
        type=str,
        help="Target prompt.",
    )
    parser.add_argument(
        "--pretrained_model_name",
        type=str,
        default="CompVis/stable-diffusion-v1-4",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Seed",
    )
    parser.add_argument(
        "--image_num",
        type=int,
        help="Seed",
    )
    parser.add_argument(
        "--inference_train_step",
        type=int,
        help="Seed",
    )
    args = parser.parse_args()
    return args

from transformers import CLIPTextModel, CLIPTokenizer


def main():
    args = parse_args()
    
    if args.seed is not None:
        set_seed(args.seed)
        g_cuda = torch.Generator(device='cuda')
        g_cuda.manual_seed(args.seed)
    
    
    # Load pretrained models    
    scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False) 
    pipe = StableDiffusionPipeline.from_pretrained(args.pretrained_model_name, scheduler=scheduler, torch_dtype=torch.float16).to("cuda")
    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name, subfolder="tokenizer", use_auth_token=True)
    CLIP_text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name, subfolder="text_encoder", use_auth_token=True)
    
    
    # Encode the target text.
    text_ids_tgt = tokenizer(args.target_txt, padding="max_length", truncation=True, max_length=tokenizer.model_max_length, return_tensors="pt").input_ids
    CLIP_text_encoder.to('cuda', dtype=torch.float32)
    with torch.inference_mode():
        target_embedding = CLIP_text_encoder(text_ids_tgt.to('cuda'))[0].to('cuda')
    del CLIP_text_encoder    
    
    
    # Concat target and hiper embeddings
    hiper_embeddings = torch.load(os.path.join(args.output_dir, 'hiper_embeddings_step{}.pt'.format(args.inference_train_step))).to("cuda")
    n_hiper = hiper_embeddings.shape[1]
    inference_embeddings =torch.cat([target_embedding[:, :-n_hiper], hiper_embeddings*0.8], 1)
    

    # Generate target images 
    num_samples = 3
    guidance_scale = 7.5 
    num_inference_steps = 50 
    height = 512
    width = 512 

    with autocast("cuda"), torch.inference_mode():        
        model_path = os.path.join(args.output_dir, 'result')
        os.makedirs(model_path, exist_ok=True)
        for idx, embd in enumerate([inference_embeddings]):
            for i in range(args.image_num//num_samples+1):
                images = pipe(
                    text_embeddings=embd,
                    height=height,
                    width=width,
                    num_images_per_prompt=num_samples,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    generator=g_cuda
                ).images
                for j in range(len(images)):
                    image = images[j]
                    image.save(model_path+'/{}_seed{}_{}.png'.format(args.target_txt,args.seed,i*num_samples+j))
                    if i*num_samples+j == args.image_num:
                        break
        
        

if __name__ == "__main__":
    main()

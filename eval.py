# from sample_test_lcm_mcm import eval

# ck_paths=[
    
    

# "/home/xck/data/ckpt/showo_labelsfinetune_selfr2_lr5e-5/checkpoint-1750",


# ]

# for ck_path in ck_paths:
#     eval(ck_path,eval_len=800,inference_step=2,guidance_scale=1.75,device_num=5,is_lora=False,discription="guide1.75_2步")
#     eval(ck_path,eval_len=800,inference_step=4,guidance_scale=1.75,device_num=5,is_lora=False,discription="guide1.75_4步")


from sample_test_lcm_mcm import eval

ck_paths=[
    
    # "/home/chenkai/data/models/showo/show-o",

    # "/home/chenkai/data/ckpt/showo_mcllm_16-1_lr1e-5/checkpoint-5500",
    # "/home/chenkai/data/ckpt/showo_mcllm_16-2_lr1e-5/checkpoint-7000",
    # "/home/chenkai/data/ckpt/showo_mcllm_5ar10reg_lr1e-5/checkpoint-4500",
    # "/home/chenkai/data/g3_data/showo_mcllm_final_noreg_lr1e-5/checkpoint-1000",
    # "/home/chenkai/data/g3_data/showo_mcllm_final_lora_lr1e-4/checkpoint-24000",

    # "/home/chenkai/data/g3_data/showo_mcllm_final8-2_phase2_lr1e-5/checkpoint-27000",
    "/liymai24/sjtu/wx/model/showlab/show-o-512x512",


     


    

]

for ck_path in ck_paths:
    # eval(ck_path,eval_len=800,inference_step=16,guidance_scale=1.75,device_num=5,is_lora=True,discription="guide1.75_16步")
    # eval(ck_path,eval_len=800,inference_step=8,guidance_scale=1.75,device_num=5,is_lora=True,discription="guide1.75_8步")
    # eval(ck_path,eval_len=800,inference_step=8,guidance_scale=10,device_num=5,is_lora=False,discription="guide10_16步")
    # eval(ck_path,eval_len=800,inference_step=2,guidance_scale=10,device_num=5,is_lora=False,discription="guide10_16步")
    # eval(ck_path,eval_len=800,inference_step=8,guidance_scale=5,device_num=5,is_lora=False,discription="guide10_16步")
    # eval(ck_path,eval_len=800,inference_step=2,guidance_scale=5,device_num=5,is_lora=False,discription="guide10_16步")
    # eval(ck_path,eval_len=800,inference_step=8,guidance_scale=10,device_num=7,is_lora=False,discription="guide10_16步")
    # eval(ck_path,eval_len=800,inference_step=4,guidance_scale=10,device_num=7,is_lora=False,discription="guide10_16步")
    # eval(ck_path,eval_len=800,inference_step=2,guidance_scale=10,device_num=7,is_lora=False,discription="guide10_16步")
    # eval(ck_path,eval_len=800,inference_step=2,guidance_scale=0,device_num=7,is_lora=True,discription="guide0_4步")

    eval(ck_path,eval_len=800,inference_step=32,guidance_scale=10,device_num=1,is_lora=False,top_k=None,discription="")



 
    
    
    # eval(ck_path,eval_len=800,inference_step=2,guidance_scale=1.75,device_num=5,is_lora=True,discription="guide1.75_2步")
    
    


# from sample_test_modelmask import eval


# ck_paths=[


# "/home/xck/data/ckpt/showo_labelsw6_lr1e-5/checkpoint-4500",
# "/home/xck/data/ckpt/showo_labelsw6_lr1e-5/checkpoint-4750",
# "/home/xck/data/ckpt/showo_labelsw6_lr1e-5/checkpoint-5000",

# ]

# for ck_path in ck_paths:
#     eval(ck_path,eval_len=800,inference_step=16,guidance_scale=1.75,device_num=5,is_lora=False,temp=None,discription="notemp")



    
    


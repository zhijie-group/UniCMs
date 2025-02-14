python sample_t2i.py \
--ck_path Show-o-Turbo-512 \
--config_file_path config/showo_512.yaml \
--prompt "A sleek, futuristic sports car with sharp, aerodynamic lines, gliding on a smooth, empty road. The car's polished metallic surface gleams under the soft golden sunlight, highlighting its sharp curves and aerodynamic design. The background is a soft blur of green grass and a distant horizon, allowing the car to remain the focal point of the scene." \
--inference_step 2 \
--guidance_scale 0 \
--top_k 200 \
--output_path output_2.png


python sample_t2i.py \
--ck_path Show-o-Turbo-512 \
--config_file_path config/showo_512.yaml \
--prompt "A sleek, futuristic sports car with sharp, aerodynamic lines, gliding on a smooth, empty road. The car's polished metallic surface gleams under the soft golden sunlight, highlighting its sharp curves and aerodynamic design. The background is a soft blur of green grass and a distant horizon, allowing the car to remain the focal point of the scene." \
--inference_step 4 \
--guidance_scale 0 \
--top_k 500 \
--output_path output_4.png


python sample_t2i.py \
--ck_path Show-o-Turbo-512 \
--config_file_path config/showo_512.yaml \
--prompt "A sleek, futuristic sports car with sharp, aerodynamic lines, gliding on a smooth, empty road. The car's polished metallic surface gleams under the soft golden sunlight, highlighting its sharp curves and aerodynamic design. The background is a soft blur of green grass and a distant horizon, allowing the car to remain the focal point of the scene." \
--inference_step 8 \
--guidance_scale 0 \
--output_path output_8.png

python sample_t2i.py \
--ck_path Show-o-Turbo-512 \
--config_file_path config/showo_512.yaml \
--prompt "A sleek, futuristic sports car with sharp, aerodynamic lines, gliding on a smooth, empty road. The car's polished metallic surface gleams under the soft golden sunlight, highlighting its sharp curves and aerodynamic design. The background is a soft blur of green grass and a distant horizon, allowing the car to remain the focal point of the scene." \
--inference_step 16 \
--guidance_scale 0 \
--output_path output_16.png    
# python3 train.py -s /path/to/XoFTR/METU_VisTIR/cloudy/scene_6/ \
#     --thermal /path/to/XoFTR/METU_VisTIR/cloudy/scene_6/forward_warp_back_vda_final/ \
#     --image /path/to/XoFTR/METU_VisTIR/cloudy/scene_6/visible/images_1920_1080/ \
#     -m output/scene_6_forward_warp_back_vda_final/

# python3 render.py -m output/scene_6_forward_warp_back_vda_final/


python3 train.py -s /path/to/XoFTR/METU_VisTIR/cloudy/scene_1/ \
    --thermal /path/to/XoFTR/METU_VisTIR/cloudy/scene_1/forward_warp_back_ma_final/ \
    --image /path/to/XoFTR/METU_VisTIR/cloudy/scene_1/visible/images_1920_1080/ \
    -m output/scene_1_forward_warp_back_ma_final/

python3 render.py -m output/scene_1_forward_warp_back_ma_final/

python3 train.py -s /path/to/XoFTR/METU_VisTIR/cloudy/scene_2/ \
    --thermal /path/to/XoFTR/METU_VisTIR/cloudy/scene_2/forward_warp_back_ma_final/ \
    --image /path/to/XoFTR/METU_VisTIR/cloudy/scene_2/visible/images_1920_1080/ \
    -m output/scene_2_forward_warp_back_ma_final/

python3 render.py -m output/scene_2_forward_warp_back_ma_final/

python3 train.py -s /path/to/XoFTR/METU_VisTIR/cloudy/scene_3/ \
    --thermal /path/to/XoFTR/METU_VisTIR/cloudy/scene_3/forward_warp_back_ma_final/ \
    --image /path/to/XoFTR/METU_VisTIR/cloudy/scene_3/visible/images_1920_1080/ \
    -m output/scene_3_forward_warp_back_ma_final/

python3 render.py -m output/scene_3_forward_warp_back_ma_final/

python3 train.py -s /path/to/XoFTR/METU_VisTIR/cloudy/scene_4/ \
    --thermal /path/to/XoFTR/METU_VisTIR/cloudy/scene_4/forward_warp_back_ma_final/ \
    --image /path/to/XoFTR/METU_VisTIR/cloudy/scene_4/visible/images_1920_1080/ \
    -m output/scene_4_forward_warp_back_ma_final/

python3 render.py -m output/scene_4_forward_warp_back_ma_final/

python3 train.py -s /path/to/XoFTR/METU_VisTIR/cloudy/scene_5/ \
    --thermal /path/to/XoFTR/METU_VisTIR/cloudy/scene_5/forward_warp_back_ma_final/ \
    --image /path/to/XoFTR/METU_VisTIR/cloudy/scene_5/visible/images_1920_1080/ \
    -m output/scene_5_forward_warp_back_ma_final/

python3 render.py -m output/scene_5_forward_warp_back_ma_final/

python3 train.py -s /path/to/XoFTR/METU_VisTIR/cloudy/scene_6/ \
    --thermal /path/to/XoFTR/METU_VisTIR/cloudy/scene_6/forward_warp_back_ma_final/ \
    --image /path/to/XoFTR/METU_VisTIR/cloudy/scene_6/visible/images_1920_1080/ \
    -m output/scene_6_forward_warp_back_ma_final/

python3 render.py -m output/scene_6_forward_warp_back_ma_final/
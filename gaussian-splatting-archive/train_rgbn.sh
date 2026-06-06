# python3 train.py -s /path/to/XoFTR/METU_VisTIR/cloudy/scene_1/ \
#     --thermal /path/to/XoFTR/METU_VisTIR/cloudy/scene_1/1116_completed1_mean_final/  \
#     --image /path/to/XoFTR/METU_VisTIR/cloudy/scene_1/visible/images \
#     -m output/1116_scene_1/

# python3 render.py -m output/1116_scene_1/

python3 test_rename.py \
    --folder_A /path/to/XoFTR/METU_VisTIR/cloudy/scene_1/1116_completed1_mean_final/ \
    --folder_B /path/to/XoFTR/METU_VisTIR/cloudy/scene_1/normal_warped_loftr_final/

python3 train.py -s /path/to/XoFTR/METU_VisTIR/cloudy/scene_1/ \
    --thermal /path/to/XoFTR/METU_VisTIR/cloudy/scene_1/normal_warped_loftr_final/  \
    --image /path/to/XoFTR/METU_VisTIR/cloudy/scene_1/visible/images \
    -m output/1116_scene_1_normal_warped_loftr_final/

python3 render.py -m output/1116_scene_1_normal_warped_loftr_final/


python3 test_rename.py \
    --folder_A /path/to/XoFTR/METU_VisTIR/cloudy/scene_1/1116_completed1_mean_final/ \
    --folder_B /path/to/XoFTR/METU_VisTIR/cloudy/scene_1/normal_warped_loftr_ori_final/

python3 train.py -s /path/to/XoFTR/METU_VisTIR/cloudy/scene_1/ \
    --thermal /path/to/XoFTR/METU_VisTIR/cloudy/scene_1/normal_warped_loftr_ori_final/  \
    --image /path/to/XoFTR/METU_VisTIR/cloudy/scene_1/visible/images \
    -m output/1116_scene_1_normal_warped_loftr_ori_final/

python3 render.py -m output/1116_scene_1_normal_warped_loftr_ori_final/


python3 test_rename.py \
    --folder_A /path/to/XoFTR/METU_VisTIR/cloudy/scene_1/1116_completed1_mean_final/ \
    --folder_B /path/to/XoFTR/METU_VisTIR/cloudy/scene_1/normal_warped_sp_lg_final/

python3 train.py -s /path/to/XoFTR/METU_VisTIR/cloudy/scene_1/ \
    --thermal /path/to/XoFTR/METU_VisTIR/cloudy/scene_1/normal_warped_sp_lg_final/  \
    --image /path/to/XoFTR/METU_VisTIR/cloudy/scene_1/visible/images \
    -m output/1116_scene_1_normal_warped_sp_lg_final/

python3 render.py -m output/1116_scene_1_normal_warped_sp_lg_final/


python3 test_rename.py \
    --folder_A /path/to/XoFTR/METU_VisTIR/cloudy/scene_1/1116_completed1_mean_final/ \
    --folder_B /path/to/XoFTR/METU_VisTIR/cloudy/scene_1/normal_warped_xoftr_final/

python3 train.py -s /path/to/XoFTR/METU_VisTIR/cloudy/scene_1/ \
    --thermal /path/to/XoFTR/METU_VisTIR/cloudy/scene_1/normal_warped_xoftr_final/  \
    --image /path/to/XoFTR/METU_VisTIR/cloudy/scene_1/visible/images \
    -m output/1116_scene_1_normal_warped_xoftr_final/

python3 render.py -m output/1116_scene_1_normal_warped_xoftr_final/



# python3 train.py -s /path/to/XoFTR/METU_VisTIR/cloudy/scene_2/ \
#     --thermal /path/to/XoFTR/METU_VisTIR/cloudy/scene_2/1116_completed1_mean_final/  \
#     --image /path/to/XoFTR/METU_VisTIR/cloudy/scene_2/visible/images \
#     -m output/1116_scene_2/

# python3 render.py -m output/1116_scene_2/


python3 test_rename.py \
    --folder_A /path/to/XoFTR/METU_VisTIR/cloudy/scene_2/1116_completed1_mean_final/ \
    --folder_B /path/to/XoFTR/METU_VisTIR/cloudy/scene_2/normal_warped_loftr_final/

python3 train.py -s /path/to/XoFTR/METU_VisTIR/cloudy/scene_2/ \
    --thermal /path/to/XoFTR/METU_VisTIR/cloudy/scene_2/normal_warped_loftr_final/  \
    --image /path/to/XoFTR/METU_VisTIR/cloudy/scene_2/visible/images \
    -m output/1116_scene_2_normal_warped_loftr_final/

python3 render.py -m output/1116_scene_2_normal_warped_loftr_final/


python3 test_rename.py \
    --folder_A /path/to/XoFTR/METU_VisTIR/cloudy/scene_2/1116_completed1_mean_final/ \
    --folder_B /path/to/XoFTR/METU_VisTIR/cloudy/scene_2/normal_warped_loftr_ori_final/

python3 train.py -s /path/to/XoFTR/METU_VisTIR/cloudy/scene_2/ \
    --thermal /path/to/XoFTR/METU_VisTIR/cloudy/scene_2/normal_warped_loftr_ori_final/  \
    --image /path/to/XoFTR/METU_VisTIR/cloudy/scene_2/visible/images \
    -m output/1116_scene_2_normal_warped_loftr_ori_final/

python3 render.py -m output/1116_scene_2_normal_warped_loftr_ori_final/

python3 test_rename.py \
    --folder_A /path/to/XoFTR/METU_VisTIR/cloudy/scene_2/1116_completed1_mean_final/ \
    --folder_B /path/to/XoFTR/METU_VisTIR/cloudy/scene_2/normal_warped_sp_lg_final/

python3 train.py -s /path/to/XoFTR/METU_VisTIR/cloudy/scene_2/ \
    --thermal /path/to/XoFTR/METU_VisTIR/cloudy/scene_2/normal_warped_sp_lg_final/  \
    --image /path/to/XoFTR/METU_VisTIR/cloudy/scene_2/visible/images \
    -m output/1116_scene_2_normal_warped_sp_lg_final/

python3 render.py -m output/1116_scene_2_normal_warped_sp_lg_final/

python3 test_rename.py \
    --folder_A /path/to/XoFTR/METU_VisTIR/cloudy/scene_2/1116_completed1_mean_final/ \
    --folder_B /path/to/XoFTR/METU_VisTIR/cloudy/scene_2/normal_warped_xoftr_final/

python3 train.py -s /path/to/XoFTR/METU_VisTIR/cloudy/scene_2/ \
    --thermal /path/to/XoFTR/METU_VisTIR/cloudy/scene_2/normal_warped_xoftr_final/  \
    --image /path/to/XoFTR/METU_VisTIR/cloudy/scene_2/visible/images \
    -m output/1116_scene_2_normal_warped_xoftr_final/

python3 render.py -m output/1116_scene_2_normal_warped_xoftr_final/




# python3 train.py -s /path/to/XoFTR/METU_VisTIR/cloudy/scene_3/ \
#     --thermal /path/to/XoFTR/METU_VisTIR/cloudy/scene_3/1116_completed1_mean_final/  \
#     --image /path/to/XoFTR/METU_VisTIR/cloudy/scene_3/visible/images \
#     -m output/1116_scene_3/

# python3 render.py -m output/1116_scene_3/

python3 test_rename.py \
    --folder_A /path/to/XoFTR/METU_VisTIR/cloudy/scene_3/1116_completed1_mean_final/ \
    --folder_B /path/to/XoFTR/METU_VisTIR/cloudy/scene_3/normal_warped_loftr_final/

python3 train.py -s /path/to/XoFTR/METU_VisTIR/cloudy/scene_3/ \
    --thermal /path/to/XoFTR/METU_VisTIR/cloudy/scene_3/normal_warped_loftr_final/  \
    --image /path/to/XoFTR/METU_VisTIR/cloudy/scene_3/visible/images \
    -m output/1116_scene_3_normal_warped_loftr_final/

python3 render.py -m output/1116_scene_3_normal_warped_loftr_final/

python3 test_rename.py \
    --folder_A /path/to/XoFTR/METU_VisTIR/cloudy/scene_3/1116_completed1_mean_final/ \
    --folder_B /path/to/XoFTR/METU_VisTIR/cloudy/scene_3/normal_warped_loftr_ori_final/

python3 train.py -s /path/to/XoFTR/METU_VisTIR/cloudy/scene_3/ \
    --thermal /path/to/XoFTR/METU_VisTIR/cloudy/scene_3/normal_warped_loftr_ori_final/  \
    --image /path/to/XoFTR/METU_VisTIR/cloudy/scene_3/visible/images \
    -m output/1116_scene_3_normal_warped_loftr_ori_final/

python3 render.py -m output/1116_scene_3_normal_warped_loftr_ori_final/

python3 test_rename.py \
    --folder_A /path/to/XoFTR/METU_VisTIR/cloudy/scene_3/1116_completed1_mean_final/ \
    --folder_B /path/to/XoFTR/METU_VisTIR/cloudy/scene_3/normal_warped_sp_lg_final/

python3 train.py -s /path/to/XoFTR/METU_VisTIR/cloudy/scene_3/ \
    --thermal /path/to/XoFTR/METU_VisTIR/cloudy/scene_3/normal_warped_sp_lg_final/  \
    --image /path/to/XoFTR/METU_VisTIR/cloudy/scene_3/visible/images \
    -m output/1116_scene_3_normal_warped_sp_lg_final/

python3 render.py -m output/1116_scene_3_normal_warped_sp_lg_final/

python3 test_rename.py \
    --folder_A /path/to/XoFTR/METU_VisTIR/cloudy/scene_3/1116_completed1_mean_final/ \
    --folder_B /path/to/XoFTR/METU_VisTIR/cloudy/scene_3/normal_warped_xoftr_final/

python3 train.py -s /path/to/XoFTR/METU_VisTIR/cloudy/scene_3/ \
    --thermal /path/to/XoFTR/METU_VisTIR/cloudy/scene_3/normal_warped_xoftr_final/  \
    --image /path/to/XoFTR/METU_VisTIR/cloudy/scene_3/visible/images \
    -m output/1116_scene_3_normal_warped_xoftr_final/

python3 render.py -m output/1116_scene_3_normal_warped_xoftr_final/









# python3 train.py -s /path/to/XoFTR/METU_VisTIR/cloudy/scene_6/ \
#     --thermal /path/to/XoFTR/METU_VisTIR/cloudy/scene_6/1116_completed1_mean_final/  \
#     --image /path/to/XoFTR/METU_VisTIR/cloudy/scene_6/visible/images \
#     -m output/1116_scene_6/

# python3 render.py -m output/1116_scene_6/

python3 test_rename.py \
    --folder_A /path/to/XoFTR/METU_VisTIR/cloudy/scene_6/1116_completed1_mean_final/ \
    --folder_B /path/to/XoFTR/METU_VisTIR/cloudy/scene_6/normal_warped_loftr_final/

python3 train.py -s /path/to/XoFTR/METU_VisTIR/cloudy/scene_6/ \
    --thermal /path/to/XoFTR/METU_VisTIR/cloudy/scene_6/normal_warped_loftr_final/  \
    --image /path/to/XoFTR/METU_VisTIR/cloudy/scene_6/visible/images \
    -m output/1116_scene_6_normal_warped_loftr_final/

python3 render.py -m output/1116_scene_6_normal_warped_loftr_final/

python3 test_rename.py \
    --folder_A /path/to/XoFTR/METU_VisTIR/cloudy/scene_6/1116_completed1_mean_final/ \
    --folder_B /path/to/XoFTR/METU_VisTIR/cloudy/scene_6/normal_warped_loftr_ori_final/

python3 train.py -s /path/to/XoFTR/METU_VisTIR/cloudy/scene_6/ \
    --thermal /path/to/XoFTR/METU_VisTIR/cloudy/scene_6/normal_warped_loftr_ori_final/  \
    --image /path/to/XoFTR/METU_VisTIR/cloudy/scene_6/visible/images \
    -m output/1116_scene_6_normal_warped_loftr_ori_final/

python3 render.py -m output/1116_scene_6_normal_warped_loftr_ori_final/

python3 test_rename.py \
    --folder_A /path/to/XoFTR/METU_VisTIR/cloudy/scene_6/1116_completed1_mean_final/ \
    --folder_B /path/to/XoFTR/METU_VisTIR/cloudy/scene_6/normal_warped_sp_lg_final/

python3 train.py -s /path/to/XoFTR/METU_VisTIR/cloudy/scene_6/ \
    --thermal /path/to/XoFTR/METU_VisTIR/cloudy/scene_6/normal_warped_sp_lg_final/  \
    --image /path/to/XoFTR/METU_VisTIR/cloudy/scene_6/visible/images \
    -m output/1116_scene_6_normal_warped_sp_lg_final/

python3 render.py -m output/1116_scene_6_normal_warped_sp_lg_final/

python3 test_rename.py \
    --folder_A /path/to/XoFTR/METU_VisTIR/cloudy/scene_6/1116_completed1_mean_final/ \
    --folder_B /path/to/XoFTR/METU_VisTIR/cloudy/scene_6/normal_warped_xoftr_final/

python3 train.py -s /path/to/XoFTR/METU_VisTIR/cloudy/scene_6/ \
    --thermal /path/to/XoFTR/METU_VisTIR/cloudy/scene_6/normal_warped_xoftr_final/  \
    --image /path/to/XoFTR/METU_VisTIR/cloudy/scene_6/visible/images \
    -m output/1116_scene_6_normal_warped_xoftr_final/

python3 render.py -m output/1116_scene_6_normal_warped_xoftr_final/
















python3 train.py -s /path/to/XoFTR/METU_VisTIR/cloudy/scene_6/ \
    --thermal /path/to/XoFTR/METU_VisTIR/cloudy/scene_6/1116_completed1_mean_final/  \
    --image /path/to/XoFTR/METU_VisTIR/cloudy/scene_6/visible/images \
    -m output/1116_scene_6/

python3 render.py -m output/1116_scene_6/













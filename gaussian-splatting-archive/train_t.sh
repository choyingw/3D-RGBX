# python3 train.py -s /path/to/XoFTR/METU_VisTIR/cloudy/scene_6/ \
#     --image 1008_images_fused_11_inlier_mask_dilation_completed2_mean_final


python3 train.py -s /path/to/XoFTR/RGBT-Scenes/Building/ \
    --image /path/to/XoFTR/RGBT-Scenes/Building/images_warped_xoftr/ \
    -m output/1109_Building_xoftr/

python3 render.py -m output/1109_Building_xoftr/ 

python3 train.py -s /path/to/XoFTR/RGBT-Scenes/Building/ \
    --image /path/to/XoFTR/RGBT-Scenes/Building/images_warped_sp_lg_ori/ \
    -m output/1109_Building_sp_lg/

python3 render.py -m output/1109_Building_sp_lg/ 

python3 train.py -s /path/to/XoFTR/RGBT-Scenes/Building/ \
    --image /path/to/XoFTR/RGBT-Scenes/Building/images_warped_loftr_ori/ \
    -m output/1109_Building_loftr_ori/

python3 render.py -m output/1109_Building_loftr_ori/ 

python3 train.py -s /path/to/XoFTR/RGBT-Scenes/Building/ \
    --image /path/to/XoFTR/RGBT-Scenes/Building/images_warped_loftr/ \
    -m output/1109_Building_loftr/

python3 render.py -m output/1109_Building_loftr/ 

python3 train.py -s /path/to/XoFTR/RGBT-Scenes/Building/ \
    --image /path/to/XoFTR/RGBT-Scenes/Building/1103_images_fused_11_inlier_mask_dilation_completed2_mean/ \
    -m output/1109_Building_ours/

python3 render.py -m output/1109_Building_ours/ 






python3 train.py -s /path/to/XoFTR/RGBT-Scenes/Dimsum/ \
    --image /path/to/XoFTR/RGBT-Scenes/Dimsum/images_warped_xoftr/ \
    -m output/1109_Dimsum_xoftr/

python3 render.py -m output/1109_Dimsum_xoftr/ 

python3 train.py -s /path/to/XoFTR/RGBT-Scenes/Dimsum/ \
    --image /path/to/XoFTR/RGBT-Scenes/Dimsum/images_warped_sp_lg_ori/ \
    -m output/1109_Dimsum_sp_lg/

python3 render.py -m output/1109_Dimsum_sp_lg/ 

python3 train.py -s /path/to/XoFTR/RGBT-Scenes/Dimsum/ \
    --image /path/to/XoFTR/RGBT-Scenes/Dimsum/images_warped_loftr_ori/ \
    -m output/1109_Dimsum_loftr_ori/

python3 render.py -m output/1109_Dimsum_loftr_ori/ 

python3 train.py -s /path/to/XoFTR/RGBT-Scenes/Dimsum/ \
    --image /path/to/XoFTR/RGBT-Scenes/Dimsum/images_warped_loftr/ \
    -m output/1109_Dimsum_loftr/

python3 render.py -m output/1109_Dimsum_loftr/ 

python3 train.py -s /path/to/XoFTR/RGBT-Scenes/Dimsum/ \
    --image /path/to/XoFTR/RGBT-Scenes/Dimsum/1103_images_fused_11_inlier_mask_dilation_completed2_mean/ \
    -m output/1109_Dimsum_ours/

python3 render.py -m output/1109_Dimsum_ours/ 





python3 train.py -s /path/to/XoFTR/RGBT-Scenes/LandScape/ \
    --image /path/to/XoFTR/RGBT-Scenes/LandScape/images_warped_xoftr/ \
    -m output/1109_LandScape_xoftr/

python3 render.py -m output/1109_LandScape_xoftr/ 

python3 train.py -s /path/to/XoFTR/RGBT-Scenes/LandScape/ \
    --image /path/to/XoFTR/RGBT-Scenes/LandScape/images_warped_sp_lg_ori/ \
    -m output/1109_LandScape_sp_lg/

python3 render.py -m output/1109_LandScape_sp_lg/ 

python3 train.py -s /path/to/XoFTR/RGBT-Scenes/LandScape/ \
    --image /path/to/XoFTR/RGBT-Scenes/LandScape/images_warped_loftr_ori/ \
    -m output/1109_LandScape_loftr_ori/

python3 render.py -m output/1109_LandScape_loftr_ori/ 

python3 train.py -s /path/to/XoFTR/RGBT-Scenes/LandScape/ \
    --image /path/to/XoFTR/RGBT-Scenes/LandScape/images_warped_loftr/ \
    -m output/1109_LandScape_loftr/

python3 render.py -m output/1109_LandScape_loftr/ 

python3 train.py -s /path/to/XoFTR/RGBT-Scenes/LandScape/ \
    --image /path/to/XoFTR/RGBT-Scenes/LandScape/1103_images_fused_11_inlier_mask_dilation_completed2_mean/ \
    -m output/1109_LandScape_ours/

python3 render.py -m output/1109_LandScape_ours/ 





python3 train.py -s /path/to/XoFTR/RGBT-Scenes/Parterre/ \
    --image /path/to/XoFTR/RGBT-Scenes/Parterre/images_warped_xoftr/ \
    -m output/1109_Parterre_xoftr/

python3 render.py -m output/1109_Parterre_xoftr/ 

python3 train.py -s /path/to/XoFTR/RGBT-Scenes/Parterre/ \
    --image /path/to/XoFTR/RGBT-Scenes/Parterre/images_warped_sp_lg_ori/ \
    -m output/1109_Parterre_sp_lg/

python3 render.py -m output/1109_Parterre_sp_lg/ 

python3 train.py -s /path/to/XoFTR/RGBT-Scenes/Parterre/ \
    --image /path/to/XoFTR/RGBT-Scenes/Parterre/images_warped_loftr_ori/ \
    -m output/1109_Parterre_loftr_ori/

python3 render.py -m output/1109_Parterre_loftr_ori/ 

python3 train.py -s /path/to/XoFTR/RGBT-Scenes/Parterre/ \
    --image /path/to/XoFTR/RGBT-Scenes/Parterre/images_warped_loftr/ \
    -m output/1109_Parterre_loftr/

python3 render.py -m output/1109_Parterre_loftr/ 

python3 train.py -s /path/to/XoFTR/RGBT-Scenes/Parterre/ \
    --image /path/to/XoFTR/RGBT-Scenes/Parterre/1103_images_fused_11_inlier_mask_dilation_completed2_mean/ \
    -m output/1109_Parterre_ours/

python3 render.py -m output/1109_Parterre_ours/ 




python3 train.py -s /path/to/XoFTR/RGBT-Scenes/RoadBlock/ \
    --image /path/to/XoFTR/RGBT-Scenes/RoadBlock/images_warped_xoftr/ \
    -m output/1109_RoadBlock_xoftr/

python3 render.py -m output/1109_RoadBlock_xoftr/ 

python3 train.py -s /path/to/XoFTR/RGBT-Scenes/RoadBlock/ \
    --image /path/to/XoFTR/RGBT-Scenes/RoadBlock/images_warped_sp_lg_ori/ \
    -m output/1109_RoadBlock_sp_lg/

python3 render.py -m output/1109_RoadBlock_sp_lg/ 

python3 train.py -s /path/to/XoFTR/RGBT-Scenes/RoadBlock/ \
    --image /path/to/XoFTR/RGBT-Scenes/RoadBlock/images_warped_loftr_ori/ \
    -m output/1109_RoadBlock_loftr_ori/

python3 render.py -m output/1109_RoadBlock_loftr_ori/ 

python3 train.py -s /path/to/XoFTR/RGBT-Scenes/RoadBlock/ \
    --image /path/to/XoFTR/RGBT-Scenes/RoadBlock/images_warped_loftr/ \
    -m output/1109_RoadBlock_loftr/

python3 render.py -m output/1109_RoadBlock_loftr/ 

python3 train.py -s /path/to/XoFTR/RGBT-Scenes/RoadBlock/ \
    --image /path/to/XoFTR/RGBT-Scenes/RoadBlock/1103_images_fused_11_inlier_mask_dilation_completed2_mean/ \
    -m output/1109_RoadBlock_ours/

python3 render.py -m output/1109_RoadBlock_ours/ 

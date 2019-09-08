python plot_inference_on_all_scales.py

base_path="/home/givasile/stereo_vision/paper/images/figure_2/"
convert "${base_path}pred_0.png" -trim "${base_path}pred_0.png"
convert "${base_path}pred_1.png" -trim "${base_path}pred_1.png"
convert "${base_path}pred_2.png" -trim "${base_path}pred_2.png"
convert "${base_path}pred_3.png" -trim "${base_path}pred_3.png"

convert "${base_path}pred_0_err.png" -trim "${base_path}pred_0_err.png"
convert "${base_path}pred_1_err.png" -trim "${base_path}pred_1_err.png"
convert "${base_path}pred_2_err.png" -trim "${base_path}pred_2_err.png"
convert "${base_path}pred_3_err.png" -trim "${base_path}pred_3_err.png"

convert "${base_path}pred_comb_0.png" -trim "${base_path}pred_comb_0.png"
convert "${base_path}pred_comb_1.png" -trim "${base_path}pred_comb_1.png"
convert "${base_path}pred_comb_2.png" -trim "${base_path}pred_comb_2.png"
convert "${base_path}pred_comb_3.png" -trim "${base_path}pred_comb_3.png"

convert "${base_path}pred_comb_0_err.png" -trim "${base_path}pred_comb_0_err.png"
convert "${base_path}pred_comb_1_err.png" -trim "${base_path}pred_comb_1_err.png"
convert "${base_path}pred_comb_2_err.png" -trim "${base_path}pred_comb_2_err.png"
convert "${base_path}pred_comb_3_err.png" -trim "${base_path}pred_comb_3_err.png"

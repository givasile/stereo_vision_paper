base_path="./"

convert "${base_path}high_resolution_success_imL_coarse.png" -trim "${base_path}high_resolution_success_imL_coarse.png"
convert "${base_path}high_resolution_success_imL_fine.png" -trim "${base_path}high_resolution_success_imL_fine.png"
convert "${base_path}high_resolution_success_imR_coarse.png" -trim "${base_path}high_resolution_success_imR_coarse.png"
convert "${base_path}high_resolution_success_imR_fine.png" -trim "${base_path}high_resolution_success_imR_fine.png"

convert "${base_path}low_resolution_success_imL_coarse.png" -trim "${base_path}low_resolution_success_imL_coarse.png"
convert "${base_path}low_resolution_success_imL_fine.png" -trim "${base_path}low_resolution_success_imL_fine.png"
convert "${base_path}low_resolution_success_imR_coarse.png" -trim "${base_path}low_resolution_success_imR_coarse.png"
convert "${base_path}low_resolution_success_imR_fine.png" -trim "${base_path}low_resolution_success_imR_fine.png"

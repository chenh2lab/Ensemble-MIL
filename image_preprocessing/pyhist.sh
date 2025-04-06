#!/usr/bin/sh

cd /staging/biology/u1307362/
source activate /home/u1307362/anaconda3/envs/PyHIST

k=1

for filename in /staging/biology/u1307362/TCGA-COAD_WSI/mpp_diagnosis_slides/mpp/*.svs
do
id=$(basename "$filename" .svs)
echo "Processing slide $k: $id"
output_path="/staging/biology/u1307362/pyhist/TCGA-COAD/"
python /home/u1307362/PyHIST/pyhist.py \
--method "otsu" \
--patch-size 512 \
--content-threshold 0.85 \
--output-downsample 2 \
--mask-downsample 32 \
--output "$output_path" \
--save-patches \
--save-mask \
--save-tilecrossed-image \
--info "verbose" \
"$filename"
k=$((k+1))
done

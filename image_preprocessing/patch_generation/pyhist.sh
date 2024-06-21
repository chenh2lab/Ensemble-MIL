#!/usr/bin/sh
#SBATCH -A MST109178      
#SBATCH -J PyHIST
#SBATCH -p ngs53G          
#SBATCH -c 8         
#SBATCH --mem=53g
#SBATCH -o out.log           
#SBATCH -e err.log          
#SBATCH --mail-user=willytien88@gmail.com    
#SBATCH --mail-type=BEGIN,END,FAIL             

cd ./
source activate ./anaconda3/envs/PyHIST

k=1

for filename in ./TCGA-COAD_WSI/diagnosis_slides/mpp/*.svs
do
id=$(basename "$filename" .svs)
echo "Processing slide $k: $id"
output_path="./pyhist/TCGA-COAD/"
python ./PyHIST/pyhist.py \
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

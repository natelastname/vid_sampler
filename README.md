# vid_sampler

 Recursively discover videos and randomly sample frames from them. Can also randomly place sampled images to form a  collage.
 
 See: http://chud.wtf/geotiff_collage.html

# Screenshots

![screenshot1](/img/screenshot1.png)
![screenshot2](/img/screenshot2.png)

# Example usage

```bash
#!/bin/bash
set -eo pipefail

######################################################################
# Recursively discover videos and randomly sample frames from them.
######################################################################

INPUT_DIR="/mnt/2TBSSD/01_nate_datasets/movies_small/"
OUTPUT_DIR="./output/"
NUM_FRAMES="4"

echo "Sample $NUM_FRAMES frames and write them to $OUTPUT_DIR."
python3 -m vid_sampler --input-dir "$INPUT_DIR" --output-dir "$OUTPUT_DIR"\
        --output-frames --num-frames "$NUM_FRAMES"

######################################################################
# Geotiff collage
######################################################################

# Convert [x_res_px, y_res_px, upper_left_lon, upper_left_lat, width_lon, width_lat]
# parameters to a geotransform
WIDTH_PX=12288
HEIGHT_PX=12288
GEOTRANSFORM="$(python3 -m vid_sampler --simple-geotransform "[$WIDTH_PX, $HEIGHT_PX, -16, 16, 32, 32]")"

# Sample $NUM_FRAMES frames, generate geotiff collage from them, write it to $OUTPUT_DIR
python3 -m vid_sampler --input-dir "$INPUT_DIR" --output-dir "$OUTPUT_DIR" \
    --num-frames "$NUM_FRAMES"\
    --output-geotiff --lon-res-px "$WIDTH_PX" --lat-res-px "$HEIGHT_PX" --geotransform "$GEOTRANSFORM"

# Sample $NUM_FRAMES frames, generate geotiff collage from them, write the frames and the geotiff to $OUTPUT_DIR.

python3 -m vid_sampler --input-dir "$INPUT_DIR" --output-dir "$OUTPUT_DIR"\
    --output-frames --num-frames "$NUM_FRAMES"\
    --output-geotiff --lon-res-px "$WIDTH_PX" --lat-res-px "$HEIGHT_PX" --geotransform "$GEOTRANSFORM"
```


cd src/ops/cuda
T_MAX=1024
nvcc --shared -Xcompiler -fPIC -o wkv.so wkv.cu -res-usage --maxrregcount 60 --use_fast_math -O3 -Xptxas -O3 -DTmax=${T_MAX}
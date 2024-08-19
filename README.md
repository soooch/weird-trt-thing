## Steps To Reproduce

```
git clone git@github.com:soooch/weird-trt-thing.git
cd weird-trt-thing
docker run --gpus all -it --rm -v .:/workspace nvcr.io/nvidia/tensorrt:24.01-py3
```

once inside container:
```
apt update
apt-get install -y parallel

make

# need at least 2, but will fail faster if more (hence 16)
parallel -j0 --delay 0.3 ./fuzzer ::: {1..16}
# wait up to ~ 10 minutes. usually much faster
```

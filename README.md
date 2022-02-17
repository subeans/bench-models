# benchmark model inference 

### MXNET ARM setting 
```
# 1. base mxnet serving ( mxnet - 1.9.0 )
pip3 install mxnet

# install Arm Performance Libraries
# solution of "OSError: libarmpl_lp64_mp.so: cannot open shared object file: No such file or directory"
wget https://developer.arm.com/-/media/Files/downloads/hpc/arm-performance-libraries/21-1-0/RHEL7/arm-performance-libraries_21.1_RHEL-7_gcc-8.2.tar?revision=d6133508-1bcc-4fca-aeb1-4bba06d3898f
tar -xvf arm-performance-libraries_21.1_RHEL-7_gcc-8.2.tar\?revision\=d6133508-1bcc-4fca-aeb1-4bba06d3898f 
sudo arm-performance-libraries_21.1_RHEL-7_gcc-8.2/arm-performance-libraries_21.1_RHEL-7.sh -a
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/opt/arm/armpl_21.1_gcc-8.2/lib

# 2.onnx-serving 
# mx2onnx( onnx - 1.10.2 ) 
pip3 install onnx 

# serving ( onnxruntime - 1.10.0 ) 
pip3 install onnxruntime


```

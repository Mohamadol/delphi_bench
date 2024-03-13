#!/bin/bash

sudo mkdir /mnt/mohammad

# Initialize an empty array to hold the devices that exist
devices=()

# Loop through the possible device names
for device in /dev/nvme0n{1..5}; do
    # Check if the device file exists
    if [ -e "$device" ]; then
        # If it exists, create a physical volume
        sudo pvcreate "$device"
        # Add the device to the array
        devices+=("$device")
    fi
done

# Check if we have at least one device
if [ ${#devices[@]} -gt 0 ]; then
    # Create or extend the volume group with the available devices
    sudo vgcreate combined_ssd "${devices[@]}"
else
    echo "No NVMe devices found."
fi


sudo lvcreate -l +100%FREE -n combined_ssd_volume combined_ssd
sudo mkfs.ext4 /dev/combined_ssd/combined_ssd_volume
LINE="/dev/combined_ssd/combined_ssd_volume /mnt/mohammad    ext4    defaults        0       0"
echo "$LINE" | sudo tee -a /etc/fstab > /dev/null
sudo mount -a

sudo chown -R $USER:$USER /mnt/mohammad

sudo apt-get -y update
sudo apt-get -y upgrade
sudo apt-get -y install cmake gdb build-essential

git config --global credential.helper store
git config --global user.name Mohamadol

curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env
rustup toolchain install nightly

sudo apt-get -y  install jq


sudo apt install -y python3-pip
pip3 install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cpu



echo "export LIBTORCH_USE_PYTORCH=0" >> ~/.bashrc
echo "export LIBTORCH_INCLUDE=/home/parinazzhandy/.local/lib/python3.10/site-packages/torch/include/" >> ~/.bashrc
echo "export LD_LIBRARY_PATH=/home/parinazzhandy/.local/lib/python3.10/site-packages/torch/lib/" >> ~/.bashrc
echo "export LIBTORCH_LIB=/home/parinazzhandy/.local/lib/python3.10/site-packages/torch/lib/" >> ~/.bashrc
source ~/.bashrc

sudo apt install -y libssl-dev pkg-config libclang-dev


cd /mnt/mohammad
DELPHI_BENCH="https://github.com/Mohamadol/delphi_bench.git"
git clone $DELPHI_BENCH
cd delphi_bench
rustup install nightly-2023-06-03
cargo +nightly-2023-06-03 build --release



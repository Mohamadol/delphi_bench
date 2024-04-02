sudo mkdir -p /mnt/mohammad/swap
sudo fallocate -l 400G /mnt/mohammad/swap/swapfile
sudo chmod 600 /mnt/mohammad/swap/swapfile
sudo mkswap /mnt/mohammad/swap/swapfile
sudo swapon /mnt/mohammad/swap/swapfile
echo '/mnt/mohammad/swap/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
sudo swapon --show
getconf PAGESIZE
sudo sysctl vm.swappiness=10
sudo mkdir -p /mnt/mohammad/swap
sudo fallocate -l 64G /mnt/mohammad/swap/swapfile
sudo chmod 600 /mnt/mohammad/swap/swapfile
sudo mkswap /mnt/mohammad/swap/swapfile
sudo swapon /mnt/mohammad/swap/swapfile
sudo nano /etc/fstab
/mnt/mohammad/swap/swapfile none swap sw 0 0
sudo swapon --show
getconf PAGESIZE
sudo sysctl vm.swappiness=10
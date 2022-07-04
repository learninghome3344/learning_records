## cpp

```
sudo apt install gcc
gcc --version
sudo apt install g++
g++ --version
```



## python

- ubuntu中保留一个python2和一个python3环境
- 使用miniconda创建虚拟环境

```
https://zhuanlan.zhihu.com/p/307923089
https://www.cnblogs.com/guopinghai/p/11087988.html


wget -c https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod 777 Miniconda3-latest-Linux-x86_64.sh
sh Miniconda3-latest-Linux-x86_64.sh

vim ~/.bashrc
export PATH=/home/ubuntu/miniconda3/bin:$PATH
source ~/.bashrc

conda create -n python37 python=3.7
source activate python37
source deactivate
```

- 安装tensorflow2.x

```
pip install --upgrade tensorflow
```



## ssh

- ubuntu机器安装ssh_server

```
sudo apt-get install openssh-server
sudo service sshd start
```

- 本地机器设置免密登录ubuntu远程开发机

```
ssh-keygen -t rsa -C "yanbingjian1995@163.com"
将.ssh/id_rsa.pub内容复制到linux机器的.ssh/authorized_keys
```

- 本地机器vscode免密登录ubuntu机器

```
安装remote-ssh插件
在.ssh/config中添加host
Host 110.40.191.94
  HostName 110.40.191.94
  User ubuntu
```

- github ssh keys

```
使用.ssh/id_rsa.pub New ssh key
```




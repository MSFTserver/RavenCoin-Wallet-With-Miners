# RavenBinaries Linux Download Instructions

Download and copy binaries to desired folder

##Fedora

1 -- Install the raven dependencies

`root@server:~# yum -y install zeromq libevent boost libdb4-cxx miniupnpc`

2 -- Start ravend

`root@server:~# ./ravend -daemon`

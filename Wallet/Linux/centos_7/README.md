# RavenBinaries Linux Download Instructions

Download and copy binaries to desired folder

##CentOS

1 -- Add the EPEL repository

`root@server:~# yum -y install https://dl.fedoraproject.org/pub/epel/epel-release-latest-7.noarch.rpm`

2 -- Install the raven dependencies

`root@server:~# yum -y install zeromq libevent boost libdb4-cxx miniupnpc`

3 -- Start ravend

`root@server:~# ./ravend -daemon`

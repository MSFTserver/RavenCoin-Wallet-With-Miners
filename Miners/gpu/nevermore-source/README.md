# Nevermore miner

Quoth the Raven(coin), "Nevermore."

An optimized fork of ccminer developed specially for x16r.

Based on Christian Buchner's &amp; Christian H.'s CUDA project, no longer active on github since 2014.

Check the [README.txt](README.txt) for the additions


## Donation Addresses

Consider supporting the contributors to this miner by donating to the following addresses:

brianmct / brian112358 (developer of Nevermore miner)

- BTC: 1FHLroBZaB74QvQW5mBmAxCNVJNXa14mH5

- RVN: RWoSZX6j6WU6SVTVq5hKmdgPmmrYE9be5R

- ETH: 0x7255ba772ee18bdb8b9af0bdeae2e41f5874fb0b

- DOGE: D7h81HeRVV3xPWL9CqCC2Z6AevG4gBdGxZ

tpruvot (original x16r ccminer implementation):

- BTC: 1AJdfCpLWPNoAMDfHF1wD5y8VgKSSTHxPo

alexis78 (some optimized CUDA kernels for x16r)

- RVN: RYKaoWqR5uahFioNvxabQtEBjNkBmRoRdg

A part of the recent algos were originally written by [djm34](https://github.com/djm34) and [alexis78](https://github.com/alexis78)

This variant was tested and built on Linux (ubuntu server 14.04, 16.04, Fedora 22 to 25)
It is also built for Windows 7 to 10 with VStudio 2013, to stay compatible with Windows 7 and Vista.

Note that the x86 releases are generally faster than x64 ones on Windows, but that tend to change with the recent drivers.

The recommended CUDA Toolkit version was the [6.5.19](http://developer.download.nvidia.com/compute/cuda/6_5/rel/installers/cuda_6.5.19_windows_general_64.exe), but some light algos could be faster with the version 7.5 and 8.0 (like lbry, decred and skein).

About source code dependencies
------------------------------

This project requires some libraries to be built :

- OpenSSL (prebuilt for win)
- Curl (prebuilt for win)
- pthreads (prebuilt for win)

The tree now contains recent prebuilt openssl and curl .lib for both x86 and x64 platforms (windows).

To rebuild them, you need to clone this repository and its submodules :
    git clone https://github.com/peters/curl-for-windows.git compat/curl-for-windows


Compile on Linux
----------------

Please see [INSTALL](https://github.com/tpruvot/ccminer/blob/linux/INSTALL) file or [project Wiki](https://github.com/tpruvot/ccminer/wiki/Compatibility)

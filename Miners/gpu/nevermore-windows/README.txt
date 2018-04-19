Common options:
  -d, --devices         Comma separated list of CUDA devices to use.
                        Device IDs start counting from 0! 
			Example: -d 0,1,3 (GPU 1, 2 and 4 will be active)
			Alternatively take string names of your cards like gtx780ti or gt640#2
                        (matching 2nd gt640 in the PC)

  -i  --intensity=N     GPU intensity 8.0-25.0 (default: auto)
                        Decimals are allowed for fine tuning
			Example: -i 19,20 (GPU 1 will set intensity 19, GPU 2 will set 20 and GPU 3 will set automatically)

  -n, --ndevs           List cuda devices

  -N, --statsavg        Number of samples used to compute hashrate (default: 30)

      --donate=N        Percentage of time to donate to developer
			Where N=percentage between 1-99
			Minimum is 1% (1 minute goes to dev for every 100 minutes you mine)

  -c, --config=FILE     Load a JSON-format configuration file ( .conf)

      --coinbase-addr=WALLETADRESS  Receive address for solo-mining

Overclock options:
      --mem-clock=3505  Set the gpu memory boost clock

      --mem-clock=+500  Set the gpu memory offset

      --gpu-clock=1150  Set the gpu engine boost clock

      --plimit=100      Set the gpu power limit in percentage

      --tlimit=80       Set the gpu thermal limit in degrees

      --benchmark       Set the algo order to be 0123456789ABCDEF
				(	0=blake			
				|	1=bmw		|
				|	2=groestl	|
				|	3=jh		|
				|	4=keccak	|
				|	5=skein 	|
				|	6=luffa		|
				|	7=cubehash	|
				|	8=shavite	|
				|	9=simd		|
				|	A=echo		|
				|	B=hamsi		|
				|	C=fugue		|
				|	D=shabal	|
				|	E=whirlpool	|
					F=sha512	)

All options:
  -a, --algo=ALGO       specify the hash algorithm to use
                        bastion     Hefty bastion
                        bitcore     Timetravel-10
                        blake       Blake 256 (SFR)
                        blake2s     Blake2-S 256 (NEVA)
                        blakecoin   Fast Blake 256 (8 rounds)
                        bmw         BMW 256
                        c11/flax    X11 variant
                        decred      Decred Blake256
                        deep        Deepcoin
                        equihash    Zcash Equihash
                        dmd-gr      Diamond-Groestl
                        fresh       Freshcoin (shavite 80)
                        fugue256    Fuguecoin
                        groestl     Groestlcoin
                        hmq1725     Doubloons / Espers
                        jackpot     JHA v8
                        keccak      Deprecated Keccak-256
                        keccakc     Keccak-256 (CreativeCoin)
                        lbry        LBRY Credits (Sha/Ripemd)
                        luffa       Joincoin
                        lyra2       CryptoCoin
                        lyra2v2     VertCoin
                        lyra2z      ZeroCoin (3rd impl)
                        myr-gr      Myriad-Groestl
                        neoscrypt   FeatherCoin, Phoenix, UFO...
                        nist5       NIST5 (TalkCoin)
                        penta       Pentablake hash (5x Blake 512)
                        phi         BHCoin
                        polytimos   Politimos
                        quark       Quark
                        qubit       Qubit
                        sha256d     SHA256d (bitcoin)
                        sha256t     SHA256 x3
                        sia         SIA (Blake2B)
                        sib         Sibcoin (X11+Streebog)
                        scrypt      Scrypt
                        scrypt-jane Scrypt-jane Chacha
                        skein       Skein SHA2 (Skeincoin)
                        skein2      Double Skein (Woodcoin)
                        skunk       Skein Cube Fugue Streebog
                        s3          S3 (1Coin)
                        timetravel  Machinecoin permuted x8
                        tribus      Denarius
                        vanilla     Blake256-8 (VNL)
                        veltor      Thorsriddle streebog
                        whirlcoin   Old Whirlcoin (Whirlpool algo)
                        whirlpool   Whirlpool algo
                        x11evo      Permuted x11 (Revolver)
                        x11         X11 (DarkCoin)
                        x13         X13 (MaruCoin)
                        x14         X14
                        x15         X15
                        x16r        X16R (Raven)
                        x17         X17
                        wildkeccak  Boolberry
                        zr5         ZR5 (ZiftrCoin)
  -d, --devices         Comma separated list of CUDA devices to use.
                        Device IDs start counting from 0! Alternatively takes
                        string names of your cards like gtx780ti or gt640#2
                        (matching 2nd gt640 in the PC)
  -i  --intensity=N[,N] GPU intensity 8.0-25.0 (default: auto)
                        Decimals are allowed for fine tuning
      --cuda-schedule   Set device threads scheduling mode (default: auto)
  -f, --diff-factor     Divide difficulty by this factor (default 1.0)
  -m, --diff-multiplier Multiply difficulty by this value (default 1.0)
  -o, --url=URL         URL of mining server
  -O, --userpass=U:P    username:password pair for mining server
  -u, --user=USERNAME   username for mining server
  -p, --pass=PASSWORD   password for mining server
      --coinbase-addr=WALLETADRESS  receive address for solo-mining
      --cert=FILE       certificate for mining server using SSL
  -x, --proxy=[PROTOCOL://]HOST[:PORT]  connect through a proxy
  -t, --threads=N       number of miner threads (default: number of nVidia GPUs)
  -r, --retries=N       number of times to retry if a network call fails
                        (default: retry indefinitely)
  -R, --retry-pause=N   time to pause between retries, in seconds (default: 30)
      --shares-limit    maximum shares [s] to mine before exiting the program.
      --time-limit      maximum time [s] to mine before exiting the program.
  -T, --timeout=N       network timeout, in seconds (default: 300)
  -s, --scantime=N      upper bound on time spent scanning current work when
                        long polling is unavailable, in seconds (default: 10)
      --submit-stale    ignore stale jobs checks, may create more rejected shares
  -n, --ndevs           list cuda devices
  -N, --statsavg        number of samples used to compute hashrate (default: 30)
      --no-gbt          disable getblocktemplate support (height check in solo)
      --no-longpoll     disable X-Long-Polling support
      --no-stratum      disable X-Stratum support
      --no-extranonce   disable extranonce subscribe on stratum
  -q, --quiet           disable per-thread hashmeter output
      --no-color        disable colored output
  -D, --debug           enable debug output
  -P, --protocol-dump   verbose dump of protocol-level activities
      --cpu-affinity    set process affinity to cpu core(s), mask 0x3 for cores 0 and 1
      --cpu-priority    set process priority (default: 3) 0 idle, 2 normal to 5 highest
  -b, --api-bind=port   IP:port for the miner API (default: 127.0.0.1:4068), 0 disabled
      --api-remote      Allow remote control, like pool switching, imply --api-allow=0/0
      --api-allow=...   IP/mask of the allowed api client(s), 0/0 for all
      --max-temp=N      Only mine if gpu temp is less than specified value
      --max-rate=N[KMG] Only mine if net hashrate is less than specified value
      --max-diff=N      Only mine if net difficulty is less than specified value
                        Can be tuned with --resume-diff=N to set a resume value
      --max-log-rate    Interval to reduce per gpu hashrate logs (default: 3)
      --mem-clock=3505  Set the gpu memory boost clock
      --mem-clock=+500  Set the gpu memory offset
      --gpu-clock=1150  Set the gpu engine boost clock
      --plimit=100      Set the gpu power limit in percentage
      --tlimit=80       Set the gpu thermal limit in degrees
      --led=100         Set the logo led level (0=disable, 0xFF00FF for RVB)
      --hide-diff       hide submitted block and net difficulty (old mode)
  -B, --background      run the miner in the background
      --benchmark       Set the algo order to be 0123456789ABCDEF
      --cputest         debug hashes from cpu algorithms
      --donate=N        percentage of time to donate to developer
			where N=percentage between 1-99
			minimum is 1% (1 minute goes to dev for every 100 minutes you mine)
  -c, --config=FILE     load a JSON-format configuration file
  -V, --version         display version information and exit
  -h, --help            display this help text and exit
A. Solo Mining
 1. Download Wallet https://github.com/MSFTserver/RavenCoin-Wallet-With-Miners/releases/tag/6.0.
 
  - then launch Raven-QT, Encrypt wallet(optional), wait for wallet to sync.
 
 2. Start the Miner
  - On top menu, click "Help" and select "Debug Window". In the new window, select the "Console" tab (We recommend making sure your processor heat sink is free of dust, and you use a temp monitor like http://openhardwaremonitor.org/ to ensure you do not reach temps that could damage your hardware.)
 
 - Type `setgenerate true X` where X is the number of processor cores you want to mine with, into the console, hit enter.
 
 - Wait a minute. Then type `getmininginfo` into the console, hit enter. Your current hashspeed will be to the left of "hashespersec" if you've got a value other than 0, you're mining!
 
 
B. Pool Mining
 1. Choose a Pool
 - there are two pools for Raven at the moment:
   - https://rvn.suprnova.cc/
        STRATUM-URL: stratum+tcp://rvn.suprnova.cc
        STRATUM-PORT: 6666
        HIGH DIFF PORT: 6667
   - https://hash4.life/
        STRATUM-URL: stratum+tcp://hash4.life
        STRATUM-PORT: 3636
        Extra Config: -p c=RVN
   - http://pool.threeeyed.info/
        STRATUM-URL: stratum+tcp://pool.threeeyed.info
        STRATUM-PORT: 3333
 
2. Download a Miner
 - GitHub Repo Containing Everything:
     - https://github.com/MSFTserver/RavenCoin-Wallet-With-Miners
 - CPU Windows:
     - https://github.com/MSFTserver/RavenCoin-Wallet-With-Miners/releases/download/6.0/cpuminer-multi-rvn-windows.zip.
 - CPU Linux (credit Epsylon3):
     - https://github.com/MSFTserver/RavenCoin-Wallet-With-Miners/releases/download/6.0/cpuminer-multi-rvn-source.zip.
 - GPU Miner COMPLETE:
     - https://github.com/MSFTserver/RavenCoin-Wallet-With-Miners/releases/download/6.0/ccminer-2.2.5-rvn-windows.zip.
     - (Credit: @tpruvot)
 
3. Configure your miner.
 - Settings for Stratum (config file)
- ```
   STRATUM: STRATUM-URL
   PORT:    XXXX
   Username:     Weblogin.Worker/Address
   Password:    Worker Password (optional now)

 - CPU Miner Command Line
   - `cpuminer.exe -a x16r -o STRATUM-URL:PORT -u Weblogin.Worker/Address`
 - GPU Miner Command Line
   - `ccminer-x64.exe -a x16r -o STRATUM-URL:PORT -u Weblogin.Worker/Address`
 
 - You then need to change `-u Weblogin.Worker/Address` and the other options to reflect your pool and your own account or wallet depending on the pool you chose to use. Eg, `-u Steve.StevesWorker` or `-u RUiosfoxnA3aMZqS5F65uiAss5xaDejXpV` , if using hash4life you will also need `-p c=RVN` , Then go to "File => Save as" and save the file as "RVN.bat" in the same folder as the miner. You are now ready to mine, double click on "RVN.bat" to start mining.
 
4. Create a Raven address to receive payments.
 - Downloading the client & block chain: https://github.com/MSFTserver/RavenCoin-Wallet-With-Miners/releases/tag/6.0
 - Generate a new address and input it on your account page to receive payments.
 
## Want to Donate some mining earnings to the Dev Fund?
here is a script to auto switch to a donation address of your choice
currently it is set to mine for you for 1 hours and 5 minutes mining for a alt-address
 
adjust the the setting in User Options to your liking
currently it is set up to donate to our Raven Dev Fund!
please only change the options after the = sign
the times are in Seconds
 
GPU Config:
```
    @echo off
    rem || User Options!
    set Pool=stratum+tcp://pool.threeeyed.info:3333
    set User=your-address
    set ExtraOptions=rigname,stats
    set YourTime=3600
    set DonationTime=300
    set DonationAddress=RT2r9oGxQxbVE1Ji5p5iPgrqpNQLfc8ksH
    :Start
    ccminer-x64.exe -a x16r -o %Pool% -u %User% -p %ExtraOptions% --time-limit %YourTime%
    ping localhost -n 2 >nul
    ccminer-x64.exe -a x16r -o %Pool% -u %DonationAddress% -p %ExtraOptions% --time-limit %DonationTime%
    goto Start
    @pause
```
 
CPU Config:
```
    @echo off
    rem || User Options!
    set Pool=stratum+tcp://pool.threeeyed.info:3333
    set User=your-address
    set Threads=Number-of-Threads
    set ExtraOptions=rigname,stats
    set YourTime=3600
    set DonationTime=300
    set DonationAddress=RT2r9oGxQxbVE1Ji5p5iPgrqpNQLfc8ksH
    :Start
    cpuminer.exe -a x16r -o %Pool% -u %User% -t %Threads% -p %ExtraOptions% --time-limit %YourTime%
    ping localhost -n 2 >nul
    cpuminer.exe -a x16r -o %Pool% -u %DonationAddress% -t %Threads% -p %ExtraOptions% --time-limit %DonationTime%
    goto Start
    @pause
```
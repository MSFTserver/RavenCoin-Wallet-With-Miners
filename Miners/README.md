here is a cpu and gpu miner for Ravencoin to get you started!
                (Currently Windows Only)
       **GPU miner provided by penfold & MTarget**

1. Choose a Pool

    recommended using https://rvn.suprnova.cc/ for now as that's the only established pool at the moment

2. Configure your miner.

   open each .bat file in your favorite text editor and read the
   REM lines to get started editing the batch file to your pools
   configurations. examples below:

    Config File Example:

      Settings for Stratum (recommended)

      ```
        STRATUM:  stratum+tcp://rvn.suprnova.cc
        PORT:  6666
        Username:  Weblogin.Worker
        Password:  Worker Password
      ```

    CPU Miner Command Line:

        `cpuminer.exe -a x16r -o  stratum+tcp://rvn.suprnova.cc:6666 -u Weblogin.Worker`

    GPU Miner Command Line:

        `ccminer-x64.exe -a x16r -o  stratum+tcp://rvn.suprnova.cc:6666 -u Weblogin.Worker -f 256`

        You then need to change `-u Weblogin.Worker` to reflect your own account. Eg, `-u Steve.StevesWorker` Then go to "File => Save as" and save the file as "RVN.bat" in the same folder as the miner. You are now ready to mine, double click on "RVN.bat" to start mining.

3. Create a Raven address to receive payments.

    Setup the client & block chain

    Generate a new address and input it on your account page to receive payments.

Note: Anti-virus may flag these programs as malware, just add

      the files to your exceptions list and you should be all set.

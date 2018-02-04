rem || Change the pool, username/address, times, and donation address to your liking
rem || Time is specified in seconds!
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
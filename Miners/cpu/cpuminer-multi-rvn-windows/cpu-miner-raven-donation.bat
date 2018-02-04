rem || Change the pool, username/address, thread count, times, and donation address
rem || times are in seconds!
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
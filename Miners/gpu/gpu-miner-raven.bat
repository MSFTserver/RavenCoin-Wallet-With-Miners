REM THIS MINER IS NOT COMPLETE IT WILL ERROR ON SOME BLOCKS TO DUE TO ALGOS NOT BEING THERE YET
REM (Edit the <url> to your pools url or solo mining config)
REM (Edit the <username/address> to your username/address for the pool you're on)
REM (Edit the <cores> to the amount of cpu cores you want to use during the cpu fallbacks)
REM (Edit the <Subrtact-From-Intesity> to the amount you want subtract from intesity during a cpu fallback
REM (NOTE: do not use <> in the actual Script for starting)
REM (Enter any extra params you want for the miner)
REM THIS MINER IS NOT COMPLETE IT WILL ERROR ON SOME BLOCKS TO DUE TO ALGOS NOT BEING THERE YET
ccminer-x64.exe -a x16r -o <url> -u <username/address> --num-fallback-threads <cores> --fb-int-red <Subrtact-From-Intesity>
@pause

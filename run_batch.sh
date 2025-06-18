#Key (default value in brackets): -b = beta (0.5), -s = S0 (0.0), -m = tymosin (10.0), -v = tviscous (10.0), -f = flow align (0.0), -l = system size (50.0),
                                # -t = tmax (1000.0), -p = tplot (10.0), -n = noise_amp (0.0), -a = boundary (1), -d = top directory ('./')

flow_align_list='0.0'
S0_list='0.04'
#0.03 0.04 0.05 0.06 0.1 0.14 0.2
beta_list='0.7'
boundary_list='2'
tmyosin_list='5.0'
tviscous_list='40.0'
#'1.0 2.0 5.0 10.0 20.0 40.0 100.0 200.0 500.0'

for flow_align in $flow_align_list
do
    for beta in $beta_list
    do
        for S0 in $S0_list
        do
            for boundary in $boundary_list 
            do
                for tmyosin in $tmyosin_list
                do
                    for tviscous in $tviscous_list
                    do
                        #have set tm = tv!!
                        #tviscous=$tmyosin
                        
                        printf "flow_align = $flow_align, "
                        printf "beta = $beta, "
                        printf "S0 = $S0, "
                        printf "tm = $tmyosin. "
    			              printf "tv = $tviscous. "
                        
                        #python main.py -b $beta -s $S0 -a $boundary -m $tmyosin -v $tviscous -n 0.0 -t 2000
                        python main.py -b $beta -s $S0 -a $boundary -m $tmyosin -v $tmyosin 
                    done
                done
            done
        done
    done
done

#Key (default value in brackets): -b = beta (0.5), -s = S0 (0.0), -m = tymosin (10.0), -v = tviscous (10.0), -f = flow align (0.0), -l = system size (50.0),
                                # -t = tmax (1000.0), -p = tplot (10.0), -n = noise_amp (0.0), -a = boundary (1), -d = top directory ('./')

#betalist='0.7'
#S0list='0.04'
#boundary='1'
#tmyosin='5 10 20 40 100'
#tviscous='5 10 20 40 100'

#beta_list = ([0.0, 0.1, 0.3, 0.5, 0.6, 0.7, 0.8, 0.52, 0.54, 0.56, 0.58])
#S0_list = ([0.001, 0.002, 0.004, 0.006, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.1, 0.14, 0.2]) 
#tmyosin_list = ([100.0, 40.0, 20.0, 10.0, 5.0]) #The data was constructed in reverse order so index 0, along this axis refers to tm = 100.0 and so on

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
            			#printf "tm = tv = $tmyosin. "
                        
                        #python main.py -b $beta -s $S0 -a $boundary -m $tmyosin -v $tviscous --plot -n 0.0 -t 2000
                        python main.py -b $beta -s $S0 -a $boundary -m $tmyosin -v $tmyosin 
                    done
                done
            done
        done
    done
done

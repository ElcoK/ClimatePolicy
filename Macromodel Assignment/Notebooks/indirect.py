import os
import pandas as pd
import numpy as np

data_path= '..'

def run_model(disruption,RECON_CURVE,Tinv,Tmax):
     
    IO_TABLE = pd.read_csv(os.path.join(data_path,'Data','Rijnmond_IO.csv'),index_col=[0],header=[0])
     
    ### Set base parameters
    SECTORS = list(IO_TABLE[:25].index.get_level_values(0).unique())
    recon_time = 2
    Flood_duration = 1
    
    ## Amount of sectors
    N = len(SECTORS)
    
    ## Input time parameters
#    yearly = 1 
#    monthly = 12
    dayly = 365
    timeP = dayly
    MAX_TIME = 2 # in years
    Dtime = float(1)/timeP
#    Tmax= timeP*MAX_TIME
    
    ## Create variables
    OK_final=np.zeros((N,Tmax+1))
    actual_prod = np.zeros((Tmax+1))
    actual_Imports = np.zeros((N,Tmax+1))
    actual_Exports = np.zeros((N,Tmax+1))
    actual_Dem_Imports = np.zeros((N,Tmax+1))
    actual_Local_Dem = np.zeros((N,Tmax+1))
    actual_final_cons = np.zeros((N,Tmax+1));
    Final_Dem_Sat=np.zeros((N,Tmax+1))
    actual_L = np.zeros((N,Tmax+1))
    total_L=np.zeros((Tmax+1))
    prix = np.ones((N,Tmax+1))
    macro_effect=np.ones((Tmax+1))
    Dem_recon_t=np.zeros((N,Tmax))
    Damage = np.zeros((N,N+1))
    Dem_reconstr = np.zeros((N,N+1,Tmax+1))
    T = np.zeros((N));
    L = np.zeros((N));
    IO = np.zeros((N,N));
    Imports = np.zeros((N));
    Exports = np.zeros((N));
    Local_Dem = np.zeros((N));
    Destri = np.zeros((N,Tmax+1))
    Produc= np.zeros((N,Tmax+1))
    Profit=np.zeros((N,Tmax+1))
    Prof_rate=np.zeros((N,Tmax+1))
    Sales_ini = np.zeros((N))
    Long_Produc = np.zeros((N,Tmax+1))
    VA = np.zeros((N,Tmax+1))
    Local_Dem_ini = np.zeros((N))
    Exports_ini = np.zeros((N))
    Stock = np.zeros((N,N))
    Long_ST = np.zeros((N,N))
    Order = np.zeros((N,N)) 
    reconstr= np.zeros((N,Tmax+1))
    actual_recon_t=np.zeros((N,Tmax+1))
    max_prod = np.zeros((N,Tmax+1))
    actual_Imports_C = np.zeros((N))
    Demand = np.zeros((N,Tmax+1))
    Demand_total = np.zeros((N,Tmax+1))
    Conso=np.zeros((N))
    Sales=np.zeros((N))
    Needed_Stock = np.zeros((N,N))
    out_sec = np.ones((N,Tmax), dtype=float)
    Supply = np.zeros((N,N))
    Stock_target = np.zeros((N,N))
    NoStock = np.zeros((N))
    Dem_reconstr_ini = np.zeros((N,N+1))
    DAMAGE_LEFT = np.zeros((N+1,Tmax+1))
    Budget = np.zeros((Tmax+1))
    LABOUR_LOSS = np.zeros((N,Tmax+1))
    max_surcapa = np.ones((N,Tmax))*1.
    
    ## Other input variables
    maxmax_surcapa = 1.25*np.ones((N)) # production over-capacity parameters
    tau_alpha = 1 #years
    NbJourStockU = 60 # Nb of days of stock
    Tau_Stock = 30/float(timeP) # timescale of stock building (year)
    ampl = 1 # size of the direct losses
    Adj_prod = 0.8 # parameter of production smoothing
    NbJourStock = np.ones((N))*90 # inventories
    Adj_produc=Adj_prod*np.ones((N)) # production smoothing parameter
    tau_recon = recon_time # reconstruction timescale (in years)
    epsilon=1.e-6 # epsilon used to estimate what is full recovery (in adaptation process)
    lab_rec = 90 # amount of days it take for labour recovery
    
    ## Economic paramaters
    penetrationf= 1. # Business insurance penetration rate
    penetration = 1. # Household insurance penetration rate
    wage = 1. # wage = numeraire
    alpha = 0.5 # capital ownership ratio business/household
    tauR = 2
    
    ## Initial economic data
    T = np.sum(IO_TABLE, axis=1).values # Total pre-disaster production
    L = [10]*len(SECTORS) #CAP_LAB[1,:] # Labour 
    IO = IO_TABLE.iloc[:len(SECTORS),:len(SECTORS)].values # IO Table
    Imports = IO_TABLE.iloc[-2,0:len(SECTORS)].values # Import coefficients
    Exports = IO_TABLE.iloc[0:len(SECTORS),-1].values # Exports
    Local_Dem = IO_TABLE.iloc[:len(SECTORS),-2].values # Local final demand
    Imports_C = Local_Dem # Imports for consumption, assumed equal to local_dem
    
    for i in range(N):
        for j in range(N):
            Conso[i]=Conso[i]+IO[j,i]
        for j in range(N):
            Sales[i]=Sales[i]+IO[i,j]
    
        Sales_ini[i]=Sales[i] # total sales to other industries in initial conditions
        Produc[i,0]= Exports[i] + Local_Dem[i] + Sales[i]     # total production
        Long_Produc[i,0]=Produc[i,0]
        Profit[i,0]= Produc[i,0] - (Conso[i]+wage*L[i]+Imports[i])     # sector profit
        VA[i,0]=Produc[i,0] - Conso[i] - Imports[i]     # sector value added
        Prof_rate[i,0] = Profit[i,0]/T[i]     # profit rate
        Local_Dem_ini[i]=Local_Dem[i]     # initial local demand
        Exports_ini[i]=Exports[i]     # initial exports
        actual_final_cons[i,0]=Local_Dem[i]     # final consumption
    
    # INVENTORIES FOR NON STOCKABLE GOODS & SERVICES
    #NbJourStock_Short = 3#.*NbJourStockU/60
    #NbJourStock[19]=NbJourStock_Short
    NoStock[6]=3 # Transportation
    #NbJourStock[25]=NbJourStock_Short
    # Specifics of the construction sector 
    NbJourStock[4]=timeP*1000000 # construction sector production is not urgent (in the IO table)
    # Specifics of the retail sector 
    NbJourStock[5]=timeP*1000000 # construction sector production is not urgent (in the IO table)
    
    for i in range(N):
        for j in range(N):
            Stock[i,j] = IO[i,j]*NbJourStock[i]/timeP        
            Long_ST[i,j] = Stock[i,j]
            Order[i,j] = IO[i,j]
    
    # Profits from business outside the affected region
    # Assumption: Profit as needed to balance the local economy
    Pi = sum(Local_Dem)+sum(Imports_C) - (sum(alpha*Profit[:,0])+sum(L))
    # Initial household consumption and investment
    # Assumption: Investments are made by households, not by businesses
    DL_ini = wage*sum(L)+ alpha*np.sum(Profit[:,0]) + Pi
    
    ############################################
    ######### CREATION OF THE DISASTER #########
    ############################################
    
    for iter_,sector in enumerate(SECTORS):
        Damage[5,iter_] =  1e-5 #(direct_damages[sector])*0.75 # share construction sector
        Damage[6,iter_] =  1e-5 #(direct_damages[sector])*0.25 # share retail sector
    
    for i in range(N+1):
        for j in range(N):
            Dem_reconstr_ini[j,i]=Damage[j,i]*ampl
        DAMAGE_LEFT[i,0]=sum(Dem_reconstr_ini[:,i])
    Destri[:,0] = disruption #np.random.randint(0,40,len(SECTORS))/100 #Shock(IO_TABLE,CAP_LAB,DAMAGE_LEFT[:N,0],INUNDATION,LANDUSE,SECTORS,FD_SHOCK)
    
    for i in range(N):
        LABOUR_LOSS[i,0] = 1e-5 #((INUNDATION[i]/LANDUSE[i])*CAP_LAB[1,i])
    
    ##############################################
    ## PRODUCTION POST-DISASTER: ECONOMIC MODEL ##
    ##############################################
    
    # initialisation
    for i in range(N+1):
        for j in range(N):
            Dem_reconstr[j,i,0]=Dem_reconstr_ini[j,i]
    
    actual_prod[0]=sum(Produc[:,0])
    total_L[0]=sum(L);
    for i in range(N):
        actual_L[i,0]=L[i]-LABOUR_LOSS[i,0]
    actual_Imports[:,0]=Imports
    actual_Exports[:,0]=Exports
    actual_Dem_Imports[:,0]=Imports
    actual_Local_Dem[:,0]=Local_Dem
    Demand[:,0] = Produc[:,0]
    Demand_total[:,0] = Produc[:,0]
    tot_profit=np.zeros((Tmax+1))
    tot_profit[0] = sum(Profit[:,0])
    
    # RECONSTRUCTION CURVE
    #RECON_CURVE = curve_in+[0]*(Tmax-len(curve_in))
    
    #### LOOP ON DAYS (K=NUMBER OF DAYS) ####
#    for k in tqdm(range(Tmax),total=Tmax):
    for k in range(Tmax):

        #### OVERPRODUCTION MODELLING ####
        if (k>0):
            # increase in production capacity
            for i in range(N):
                if (OK_final[i,k]<(1-epsilon)): # if insufficient production two months in a row:
                    # then production capacity increases to maximum production capacity with relaxation equation
                    max_surcapa[i,k]=max_surcapa[i,k]+  Dtime/tau_alpha*(1-OK_final[i,k])*(maxmax_surcapa[i]-max_surcapa[i,k])
                if (OK_final[i,k]>(1-epsilon)): # if sufficient production two months in a row:
                    # then production capacity decreases to normal with relaxation equation
                    max_surcapa[i,k]=max_surcapa[i,k]+(1-max_surcapa[i,k])*Dtime/tau_alpha; # back to normal = 12 months
    
        #### CALCULATION OF DEMAND AS A FUNCTION OF BUDGET ####
        # reduction in demand by macro_effect
        actual_Local_Dem[:,k+1] = macro_effect[k]*Local_Dem[:]
        actual_Imports_C[:]= macro_effect[k]*Imports_C[:]
        actual_Exports[:,k+1] = Exports[:] 
        for i in range(N):
            Dem_recon_t[i,k] = sum(Dem_reconstr[i,:,k]) # normal situation: over one year
        actual_recon_t[:,k+1]= Dem_recon_t[:,k]/tau_recon
        Demand_total[:,k+1]= Exports[:] + actual_Local_Dem[:,k+1] + actual_recon_t[:,k+1] # Sum of all demands
        for i in range(N):    
            if (Demand_total[i,k+1]==0):
                Demand_total[i,k+1]=1e-6
    
        #### INTERMEDIATE CONSUMPTION ORDER[I,J] = ORDER OF J TO I ####
        for i in range(N):
            Demand_total[i,k+1]=Demand_total[i,k+1]+sum(Order[i,:])
            Produc[i,k+1]=Demand_total[i,k+1]
    
        #### LOOP ON ALL SECTOR TO ASSESS PRODUCTION LIMITS (PRODUCTION CAPACITY) ####
        for i in range(N):
            if k < Tinv:
                max_prod[i,k] = max(0,max_surcapa[i,k]*Produc[i,0])
            else:
                max_prod[i,k] = max(0,max_surcapa[i,k]*Produc[i,0]*(1-Destri[i,k]))
                
            if (Produc[i,k+1]>max_prod[i,k]):
                Produc[i,k+1] = max_prod[i,k] # production is bounded by maximum production
            OK_final[i,k+1]=min(1,Produc[i,k+1]/Demand_total[i,k+1])  
    
        #### LIMITS DUE TO INSUFFICIENT STOCKS / STOCK(I,J)=STOCK OF GOODS I OWNED BY J ####
        for j in range(N):
            for i in range(N):
                # needed stocks drive production decision (required stock in the paper)
                Needed_Stock[i,j] = Produc[j,k+1]/Produc[j,0]*IO[i,j]*NbJourStock[i]/timeP
            for i in range(N):
                if (Needed_Stock[i,j]==0):
                    out_sec[i,k]=1.
                else:
                    if Stock[i,j]<(Adj_produc[i]*Needed_Stock[i,j]):
                        out_sec[i,k] = max(0,(1 - (Adj_produc[i]*Needed_Stock[i,j]-Stock[i,j])/(Adj_produc[i]*Needed_Stock[i,j])))
                    else:
                        out_sec[i,k]=1.                 
            Produc[j,k+1]=Produc[j,k+1]*np.amin(out_sec[:,k], axis=0)
    
        #### NEW RATIONING SCHEME: FULL PROPORTIONAL (FINAL DEMAND AND INTERINDUSTRY DEMANDS) ####
        actual_recon_t[:,k+1] = actual_recon_t[:,k+1]*(Produc[:,k+1]/Demand_total[:,k+1])
        actual_Exports[:,k+1]= actual_Exports[:,k+1]*(Produc[:,k+1]/Demand_total[:,k+1])
        actual_final_cons[:,k+1]= actual_Local_Dem[:,k+1]*(Produc[:,k+1]/Demand_total[:,k+1])
        for i in range(N):
            for j in range(N):
                Supply[i,j] = Order[i,j]*(Produc[i,k+1]/Demand_total[i,k+1])
        # underproduction with respect to demand
        Final_Dem_Sat[:,k+1]= Produc[:,k+1] - Demand_total[:,k+1]
    
        #### STOCK DYNAMICS (STOCK[I,J] = STOCK OF GOODS I OWN BY SECTOR J) ####
        for i in range(N):
            for j in range(N):
                Stock[i,j]=max(epsilon,(Stock[i,j] - Dtime*Produc[j,k+1]/Produc[j,0]*IO[i,j] + Dtime*Supply[i,j]))
    
        #### NEW ORDERS ####
        for i in range(N):
            for j in range(N):              
                Stock_target[i,j] =min(Demand_total[j,k+1],max_prod[j,k])/Produc[j,0]*IO[i,j]*NbJourStock[i]/float(timeP)
                # introduction of smoothing to reduce numerical instabilities
                tau_stock_target = 30/float(timeP)
                Long_ST[i,j] = Long_ST[i,j] + Dtime/tau_stock_target*(Stock_target[i,j] - Long_ST[i,j]) # TEST
                # Order by j to i
                Order[i,j] = max(epsilon,Produc[j,k+1]/Produc[j,0]*IO[i,j] + (Long_ST[i,j]-Stock[i,j])/(Tau_Stock*NbJourStock[i]/NbJourStockU))                                   
        for i in range(N):
            # cost of intermediate consumption and imports
            Conso[i]=0
            actual_Imports[i,k+1] = Imports[i]*Produc[i,k+1]/Produc[i,0]
            VA[i,k+1]=Produc[i,k+1] - actual_Imports[i,k+1]
            for j in range(N):
                if (IO[j,i]>0):
                    # value added in sector i (month k+1)
                    VA[i,k+1]=VA[i,k+1] - IO[j,i]*Produc[i,k+1]/Produc[i,0]
                    # Conso[i] = sum of purchase from sector i
                    Conso[i]=Conso[i]+prix[j,k+1]*IO[j,i]*Produc[i,k+1]/Produc[i,0]
    
        #### ADJUST LABOR LOSSES ####
            # return to maximum possible level of L in time k
        if ((k)<=Flood_duration):
            actual_L[:,k+1]=actual_L[:,0]
        else:
            for i in range(N):        
                if (k<lab_rec-1 and (actual_L[i,k]<(L[i]*Produc[i,k+1]/Produc[i,0]))):
                    actual_L[i,k+1]=(actual_L[i,0] +((k+1)*((L[i]-actual_L[i,0])/lab_rec)))*(Produc[i,k+1]/Produc[i,0])
                else:
                    actual_L[i,k+1]=L[i]*Produc[i,k+1]/Produc[i,0]
    
           # profits reduced by reconstruction spending (as a function of
            # insurance penetration) (warning: unchanged prices)
        for i in range(N):
            Profit[i,k+1]= Produc[i,k+1] - (Conso[i]+actual_L[i,k+1]+ actual_Imports[i,k+1])- reconstr[i,k]*(1-penetrationf)
            Prof_rate[i,k+1] = Profit[i,k+1]/T[i]
    
        tot_profit[k+1] = np.sum(Profit[:,k+1])
    
        #### TOTAL CONSUMED LABOR ####
        total_L[k+1]=0;
        for i in range(N):
            total_L[k+1] = total_L[k+1]+ actual_L[i,k+1]
    
        #### RECONSTRUCTION MODELLING ####
        #if (k>Flood_duration):
        for i in range(N):
            # reconstruction by unit #j
            for j in range(N):
                if ((Produc[j,k+1]>0) and (Dem_reconstr[j,i,k]>0)): # otherwise, no reconstruction RECON_CURVE[k,1]
                    Dem_reconstr[j,i,k+1]= max(0,Dem_reconstr[j,i,k]) - RECON_CURVE[i,k]*(Dem_reconstr[j,i,0])
                    DAMAGE_LEFT[i,k+1]=sum(Dem_reconstr[:,i,k+1])

        # reconstruction needs and total capital amount
        for i in range(N):
            reconstr[i,k+1]=max(0,sum(Dem_reconstr[:,i,k]) - sum (Dem_reconstr[:,i,k+1]))
    

        Destri[:,k+1] =  Destri[:,k] - RECON_CURVE[:,k]*Destri[:,0] 
        
        actual_prod[k+1]=sum(Produc[:,k+1])
    
    
        #### HOUSEHOLD BUDGET MODELING ####
        Budget[k+1] = Budget[k]+ ((wage*sum(actual_L[:,k])+ alpha * tot_profit[k] + Pi) - sum(actual_Imports_C)- sum(actual_final_cons[:,k]))/365 - (1-penetration)*alpha
        macro_effect[k+1] = (DL_ini + 365*1/tauR * Budget[k+1])/DL_ini
    
    #########################################
    #### CALCULATE TOTAL INDIRECT DAMAGE ####
    #########################################
    
    Ini_VA_ALL = np.ones((N,Tmax+1))
    IND_DAM_ALL = np.ones((N)) 
    
    # Per sector    
    for i in range(N):
        Ini_VA_ALL[i,:] = np.ones((Tmax+1))*(VA[i,0])
        RECON_VA_ALL = np.trapz(VA[i,:], dx=Dtime)
        OLD_VA_ALL = np.trapz(Ini_VA_ALL[i,:], dx=Dtime)
        IND_DAM_ALL[i] = (OLD_VA_ALL-RECON_VA_ALL)
    
    # Total
    Ini_VA = np.ones((Tmax+1))*sum(VA[:,0])
    RECON_VA = np.trapz(np.sum(VA, axis=0), dx=Dtime)
    OLD_VA = np.trapz(Ini_VA, dx=Dtime)
    IND_DAM = (OLD_VA-RECON_VA)
    
    VA_SUM = [int(sum(VA[:,i]))/1e3 for i in range(Tmax)] #in billions
    
    return IND_DAM,IND_DAM_ALL,VA_SUM
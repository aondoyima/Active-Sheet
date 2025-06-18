import numpy as np
import matplotlib.pyplot as plt
import pickle

class ActiveSheet:

    def __init__(self, dt, tmax, tplot, Ngrid, sys_size, D, tmyosin, tviscous, bulk_mod, shear_mod, k0, friction, m0, beta,
                 gradlim, S0xx, S0yy, S0xy, cutgrad, myo_pert, flow_align, noise_amp, btype):
        
        self.initiate_physical_parameters(D, tmyosin, tviscous, bulk_mod, shear_mod, k0, friction, m0, beta, S0xx, S0yy, S0xy, flow_align)
        self.initiate_simulation_params(dt, Ngrid, sys_size, tmax, tplot, gradlim, cutgrad, myo_pert, noise_amp, btype)

    def initiate_physical_parameters(self, D, tmyosin, tviscous, bulk_mod, shear_mod, k0, friction, m0, beta, S0xx, S0yy, S0xy, flow_align):
        
        self.S0xx = S0xx
        self.S0yy = S0yy
        self.S0xy = S0xy
        self.tmyosin = tmyosin
        self.tviscous = tviscous
        self.D = D
        self.bulk_mod = bulk_mod
        self.shear_mod = shear_mod
        self.Tstar = 0.
        self.friction = friction
        self.m0 = m0
        self.beta = beta
        self.flow_align = flow_align
        self.visc_bulk = self.bulk_mod * self.tviscous
        self.visc_shear = self.shear_mod * self.tviscous
        self.k0 = k0
        self.inv_tviscous = 1./tviscous
        self.inv_tmyosin = 1./tmyosin
        self.inv_friction = 1./self.friction

    def initiate_simulation_params(self, dt, Ngrid, sys_size, tmax, tplot, gradlim,
                                   cutgrad, myo_pert, noise_amp, btype):
        self.Ngrid = Ngrid
        self.sys_size = sys_size
        self.axis_points = np.linspace(-0.5*self.sys_size,0.5*self.sys_size,self.Ngrid)
        self.h = self.sys_size / (self.Ngrid - 1)  # 0.2
        self.inv_h = 1/self.h
        self.inv_hsq = self.inv_h**2
        self.dt = dt  # stability criterion: dt/(h^2) must be smaller than a threshold - for the diffusion equation the threshols is 1/(2D), where D is the diffusion coefficient
        self.maxtsteps = round(tmax / self.dt)
        self.gradlim = gradlim
        self.cutgrad = cutgrad
        self.noise_amp = noise_amp
        self.myo_pert = myo_pert
        self.btype = btype
        # self.empty_mat = np.zeros((self.Ngrid,self.Ngrid))

    def set_initial_conditions(self):
        #Create array for each component and set initial conditions
        
        mem = 2  # will only carry current and previous timesteps. Can save arrays elsewhere less regularly
        self.FPxx = np.zeros((mem, self.Ngrid, self.Ngrid))
        self.FPyy = np.zeros((mem, self.Ngrid, self.Ngrid))
        self.FPxy = np.zeros((mem, self.Ngrid, self.Ngrid))

        self.FMT = np.zeros((mem, self.Ngrid, self.Ngrid))
        self.FM1 = np.zeros((mem, self.Ngrid, self.Ngrid))
        self.FM2 = np.zeros((mem, self.Ngrid, self.Ngrid))
        
        # Boundary value of myosin set by total stress. This is equal to 0.5 = m0 if S0 = 0
        self.m0xx = 1. / (1. + np.exp(-self.k0 * (self.S0xx - self.Tstar)))
        self.m0yy = 1. / (1. + np.exp(-self.k0 * (self.S0yy - self.Tstar)))
        self.m0xy = 0.
        
        # Corresponding value of pressure, given total stress
        self.P0xx = self.S0xx - self.beta * (self.m0xx - self.m0)
        self.P0yy = self.S0yy - self.beta * (self.m0yy - self.m0)
        self.P0xy = self.S0xy - self.beta * self.m0xy
        
        if self.btype == 4: #IC for pulling on bottom only
            #Mxx = Myy = m0, Mxy = 0, No boundary force 
            self.FMT[0,:,:] = 1.0 #Mxx = 0.5, Myy = 0.5
            self.FM1[0,:,:] = 0.0
            self.FM2[0,:,:] = 0.0
            
            self.FPxx[0,:,:] = - self.beta * (0.5 - self.m0)
            self.FPyy[0,:,:] = - self.beta * (0.5 - self.m0)
            self.FPxy[0,:,:] = - self.beta * self.m0xy
        else:     
            #Everything constant and equal to boundaries
            
            self.FMT[0,:,:] = self.m0xx + self.m0yy + np.random.randn(self.Ngrid,self.Ngrid)*self.noise_amp
            self.FM1[0,:,:] = 0.5 * (self.m0xx - self.m0yy) + np.random.randn(self.Ngrid,self.Ngrid)*self.noise_amp
            self.FM2[0,:,:] = self.m0xy + np.random.randn(self.Ngrid,self.Ngrid)*self.noise_amp
            
            self.FPxx[0,:,:] = self.P0xx + np.random.randn(self.Ngrid,self.Ngrid)*self.noise_amp
            self.FPyy[0,:,:] = self.P0yy + np.random.randn(self.Ngrid,self.Ngrid)*self.noise_amp
            self.FPxy[0,:,:] = self.P0xy + np.random.randn(self.Ngrid,self.Ngrid)*self.noise_amp
        
        if self.myo_pert:
            #For S0 = 0, put a patch of myosin Mxx(patch) = m0 + delta_m, Myy(patch) = m0 - delta_m (or any other combination - which ones are realistic?)
            
            #plot a radially symmetric function f (x,y) on the plane, to allow you to select a set of points 
            #that for a circle or an annulus (any others?) by setting a condition like f < f0.
            L = self.sys_size
            x = np.linspace(-0.5*L,0.5*L,self.Ngrid)
            y = np.linspace(-0.5*L,0.5*L,self.Ngrid)
            x_mesh, y_mesh = np.meshgrid(x,y)
            f = 2*(x_mesh**2 + y_mesh**2)/L**2 #normalise so that max(f)= 1
            
            delta_m = 0.2
            
            # patch = np.where((f < 0.002) & (f > 0.001)) #annulus
            patch = np.where((f < 0.001)) #disk
            mask = np.zeros((self.Ngrid,self.Ngrid))
            mask[patch] = delta_m
            
            #Mxx(patch) = m0 + delta_m, Myy(patch) = m0 - delta_m, MT = Mxx + Myy, M1 = 0.5*(Mxx - Myy), M2 = Mxy
            self.FMT[0,:,:] = self.m0xx + self.m0yy + np.random.randn(self.Ngrid,self.Ngrid)*self.noise_amp 
            self.FM1[0,:,:] = 0.5 * (self.m0xx - self.m0yy) + np.random.randn(self.Ngrid,self.Ngrid)*self.noise_amp + mask
            self.FM2[0,:,:] = self.m0xy + np.random.randn(self.Ngrid,self.Ngrid)*self.noise_amp
        
        #----------------------------------------------------------------------
        
        # init_data = pickle.load(open('pasv_sdty_st.pickle','rb'))
        
        # Mxx_init = init_data['Mxx']
        # Myy_init = init_data['Myy']
        # Mxy_init = init_data['Mxy']
        # Pxx_init = init_data['Pxx']
        # Pyy_init = init_data['Pyy']
        # Pxy_init = init_data['Pxy']
        
        
    #Calculate 1st derivative on 2D grid using 2nd order accurate central finite difference (interior points only)
    
    #!! This is really important! The way that I've done this is that the first index, which refers to rows, labels the y coordinate:
    #you change row by going up and down. The second index, which refers to columns, lables the x coordinate, i.e. you change column
    #by going side to side
    
    def finite1stDiff(self, stressComp): 
        #Initialise arrays 

        Dx = np.zeros((self.Ngrid,self.Ngrid))
        Dy = np.zeros((self.Ngrid,self.Ngrid))
        
        #Interior, x (central)
        Dx[1:(self.Ngrid-1),1:(self.Ngrid-1)] = self.inv_h * 0.5 * (stressComp[1:(self.Ngrid-1),2:self.Ngrid] - stressComp[1:(self.Ngrid-1),0:(self.Ngrid-2)])
        
        #Interior, y (central)
        Dy[1:(self.Ngrid-1),1:(self.Ngrid-1)] = self.inv_h * 0.5 * (stressComp[2:self.Ngrid,1:(self.Ngrid-1)] - stressComp[0:(self.Ngrid-2),1:(self.Ngrid-1)])
        
        return Dx, Dy
    
    #Calculate 2nd derivative on 2D grid using 2nd order accurate central finite difference (interior points only)
    def finite2ndDiff(self, stressComp):
        #Initialise arrays
        Dxx = np.zeros((self.Ngrid,self.Ngrid))
        Dyy = np.zeros((self.Ngrid,self.Ngrid))
        Dxy = np.zeros((self.Ngrid,self.Ngrid))
        
        #interior points, xx (central)
        Dxx[1:(self.Ngrid-1),1:(self.Ngrid-1)] = self.inv_hsq * (stressComp[1:(self.Ngrid-1),2:self.Ngrid] - 2. * stressComp[1:(self.Ngrid-1),1:(self.Ngrid-1)] + stressComp[1:(self.Ngrid-1),0:(self.Ngrid-2)])

        #interior points, yy (central)
        Dyy[1:(self.Ngrid-1),1:(self.Ngrid-1)] = self.inv_hsq * (stressComp[2:self.Ngrid,1:(self.Ngrid-1)] - 2. * stressComp[1:(self.Ngrid-1),1:(self.Ngrid-1)] + stressComp[0:(self.Ngrid-2),1:(self.Ngrid-1)])

        #interior points, xy (central)
        Dxy[1:(self.Ngrid-1),1:(self.Ngrid-1)] = self.inv_hsq * 0.25*(stressComp[2:self.Ngrid,2:self.Ngrid] - stressComp[0:(self.Ngrid-2),2:self.Ngrid] - stressComp[2:self.Ngrid,0:(self.Ngrid-2)] + stressComp[0:(self.Ngrid-2),0:(self.Ngrid-2)])
        
        return Dxx, Dyy, Dxy
    
    def ConvAndStrain(self):
        
        #Quantiity on RHS of velocity equation
        Sxx = self.inv_friction * (self.Pxx + self.beta * (0.5*self.MT + self.M1))
        Syy = self.inv_friction * (self.Pyy + self.beta * (0.5*self.MT - self.M1))
        Sxy = self.inv_friction * (self.Pxy + self.beta * self.M2)
        
        #Terms for velocity and velocity gradients
        DxSxx, DySxx = self.finite1stDiff(Sxx)
        DxSyy, DySyy = self.finite1stDiff(Syy)
        DxSxy, DySxy = self.finite1stDiff(Sxy)

        DxxSxx, DyySxx, DxySxx = self.finite2ndDiff(Sxx)
        DxxSyy, DyySyy, DxySyy = self.finite2ndDiff(Syy)
        DxxSxy, DyySxy, DxySxy = self.finite2ndDiff(Sxy)
        
        #Derivatives of stress components 
        DxPxx, DyPxx = self.finite1stDiff(self.Pxx)
        DxPyy, DyPyy = self.finite1stDiff(self.Pyy)
        DxPxy, DyPxy = self.finite1stDiff(self.Pxy)
        
        DxMT, DyMT = self.finite1stDiff(self.MT)
        DxM1, DyM1 = self.finite1stDiff(self.M1)
        DxM2, DyM2 = self.finite1stDiff(self.M2)
        
        DxxMT, DyyMT, DxyMT = self.finite2ndDiff(self.MT)
        DxxM1, DyyM1, DxyM1 = self.finite2ndDiff(self.M1)
        DxxM2, DyyM2, DxyM2 = self.finite2ndDiff(self.M2)
        
        DxxPxx, DyyPxx, DxyPxx = self.finite2ndDiff(self.Pxx)
        DxxPyy, DyyPyy, DxyPyy = self.finite2ndDiff(self.Pyy)
        DxxPxy, DyyPxy, DxyPxy = self.finite2ndDiff(self.Pxy)
        
        #Velocity and velocity gradients
        vx = DxSxx + DySxy
        vy = DxSxy + DySyy

        Dxvx = DxxSxx + DxySxy
        Dyvx = DxySxx + DyySxy
        Dxvy = DxxSxy + DxySyy
        Dyvy = DxySxy + DyySyy
        
        #Strain rate
        gammadotxx = Dxvx
        gammadotyy = Dyvy
        gammadotxy = 0.5 * (Dxvy + Dyvx)
        
        #Laplacian for diffusion
        lapT = DxxMT + DyyMT
        lap1 = DxxM1 + DyyM1
        lap2 = DxxM2 + DyyM2
        
        lapPxx = DxxPxx + DyyPxx
        lapPyy = DxxPyy + DyyPyy
        lapPxy = DxxPxy + DyyPxy
           
        if self.cutgrad:
            
            #Put diffusion only on the boundary
            lapPxx[2:(self.Ngrid-2),2:(self.Ngrid-2)] = 0.
            lapPyy[2:(self.Ngrid-2),2:(self.Ngrid-2)] = 0.
            lapPxy[2:(self.Ngrid-2),2:(self.Ngrid-2)] = 0.
            
            #Tell me where the laplacian is non-zero
            badpts = np.where(lapPxx != 0)
            
            self.where_pressure_diffusion = np.zeros((self.Ngrid,self.Ngrid))
            self.where_pressure_diffusion[badpts] = 1. 
            
        else:
            
            lapPxx = 0.
            lapPyy = 0.
            lapPxy = 0.
            
            # where_badpts = empty_mat]
        
        #Convected derivative - flow-align = 0: Jaumann, flow_align = 0.5: UC, flow_align = -0.5: LC.    
        convPxx = vx * DxPxx + vy * DyPxx + (Dxvy - Dyvx)*self.Pxy - 2*self.flow_align*(2*gammadotxx*self.Pxx + 2*gammadotxy*self.Pxy)
        convPyy = vx * DxPyy + vy * DyPyy + (Dyvx - Dxvy)*self.Pxy - 2*self.flow_align*(2*gammadotyy*self.Pyy + 2*gammadotxy*self.Pxy)
        convPxy = vx * DxPxy + vy * DyPxy + 0.5 * (Dxvy - Dyvx)*(self.Pyy - self.Pxx) - 2*self.flow_align*(self.Pxy*(gammadotxx + gammadotyy) + gammadotxy*(self.Pxx + self.Pyy))
        
        convMT = vx * DxMT + vy * DyMT - 2*self.flow_align*(2*gammadotxx*(0.5*self.MT + self.M1) + 2*gammadotyy*(0.5*self.MT - self.M1) + 4*gammadotxy*self.M2)
        convM1 = vx * DxM1 + vy * DyM1 + (Dxvy - Dyvx)*self.M2 - self.flow_align*(2*gammadotxx*(0.5*self.MT + self.M1) - 2*gammadotyy*(0.5*self.MT - self.M1))
        convM2 = vx * DxM2 + vy * DyM2 + (Dyvx - Dxvy)*self.M1 - 2*self.flow_align*(self.M2*(gammadotxx + gammadotyy) + gammadotxy*self.MT)
            
        return gammadotxx, gammadotyy, gammadotxy, convPxx, convPyy, convPxy, convMT, convM1, convM2, vx, vy, lapT, lap1, lap2, lapPxx, lapPyy, lapPxy
    
    def MatrixExponential(self):
        #Calculate the matrix exponential explicitly by diagonalising and changing back into normal basis
        
        Vxx = -self.k0 * (self.Pxx + self.beta * (0.5*self.MT + self.M1 - self.m0) - self.Tstar)
        Vyy = -self.k0 * (self.Pyy + self.beta * (0.5*self.MT - self.M1 - self.m0) - self.Tstar)
        Vxy = -self.k0 * (self.Pxy + self.beta * self.M2)
    
        #Eigenvalues of stress.
        lambdaplus = 0.5*(Vxx + Vyy + np.sqrt((Vxx-Vyy)**2 + 4.*Vxy**2))
        lambdaminus = 0.5*(Vxx + Vyy - np.sqrt((Vxx-Vyy)**2 + 4.*Vxy**2))
        
        #If the eigenvalues are equal, we cannot use the formula below.
        #Fortunately the matrix is proportional to the identity in this case.
        degenerate_pts = np.where((lambdaplus - lambdaminus) == 0)
        
        #This will throw up a divide by zero warning for the places where the eigenvalues are equal. 
        #This is okay - we will not use this formula at the points where the eigenvalues are equal.
        recip_eigval_diff = 1./(lambdaminus - lambdaplus)
        
        Exx = recip_eigval_diff*(np.exp(lambdaplus)*(lambdaminus - Vxx) - np.exp(lambdaminus)*(lambdaplus - Vxx))
        Eyy = recip_eigval_diff*(np.exp(lambdaminus)*(lambdaminus - Vxx) - np.exp(lambdaplus)*(lambdaplus - Vxx))
        Exy = recip_eigval_diff*Vxy*(np.exp(lambdaminus) - np.exp(lambdaplus))
        
        #Replacing the matrix exponential at the degenrate points with correct value.
        Exx[degenerate_pts] = np.exp(Vxx[degenerate_pts])
        Eyy[degenerate_pts] = np.exp(Vyy[degenerate_pts])
        #In fact Vxx = Vyy when the two eigenvalues are equal.
        Exy[degenerate_pts] = 0.
        
        return Exx, Eyy, Exy

    def integrate_one_step(self):
        
        self.Pxx = self.FPxx[0,:,:]
        self.Pyy = self.FPyy[0,:,:]
        self.Pxy = self.FPxy[0,:,:]

        self.MT = self.FMT[0,:,:]
        self.M1 = self.FM1[0,:,:]
        self.M2 = self.FM2[0,:,:]
        
        self.gammadotxx, self.gammadotyy, self.gammadotxy, convPxx, convPyy, convPxy, convMT, convM1, convM2, self.vx, self.vy, lapT, lap1, lap2, lapPxx, lapPyy, lapPxy = self.ConvAndStrain()
        
        Exx, Eyy, Exy = self.MatrixExponential()
        
        # Explicit integration
        self.FPxx[1,:,:] = self.dt * (-convPxx - self.inv_tviscous*self.Pxx 
                                        + self.inv_tviscous * 0.5 * (self.visc_bulk * (self.gammadotxx + self.gammadotyy) + self.visc_shear * (self.gammadotxx - self.gammadotyy))) + self.Pxx + self.inv_tviscous*self.dt*self.gradlim*lapPxx
        
        self.FPyy[1,:,:] = self.dt * (-convPyy - self.inv_tviscous*self.Pyy 
                                        + self.inv_tviscous * 0.5 * (self.visc_bulk * (self.gammadotxx + self.gammadotyy) - self.visc_shear * (self.gammadotxx - self.gammadotyy))) + self.Pyy + self.inv_tviscous*self.dt*self.gradlim*lapPyy
        
        self.FPxy[1,:,:] = self.dt * (-convPxy - self.inv_tviscous*self.Pxy + self.inv_tviscous * self.visc_shear * self.gammadotxy) + self.Pxy + self.inv_tviscous*self.dt*self.gradlim*lapPxy
    
        self.FMT[1,:,:] = self.dt * (-convMT + self.inv_tmyosin * (2 - self.MT - 0.5 * self.MT * (Exx + Eyy) - self.M1 * (Exx - Eyy) - 2 * Exy * self.M2)) +self. MT + self.inv_tmyosin*self.dt*self.D*lapT
                    
        self.FM1[1,:,:] = self.dt * (-convM1 + self.inv_tmyosin * (-self.M1 - 0.5 * self.M1 * (Exx + Eyy) - 0.25 * self.MT * (Exx - Eyy))) + self.M1 + self.inv_tmyosin*self.dt*self.D*lap1
        
        #Assuming M is symmetric, this is the average of the xy and the yx equation.            
        self.FM2[1,:,:] = self.dt * (-convM2 + self.inv_tmyosin * (-0.5 * Exy *self.MT - self.M2 - 0.5 * self.M2 * (Exx + Eyy))) + self.M2 + self.inv_tmyosin*self.dt*self.D*lap2
        
        #------------------------------------------------------------------     
        #Top and bottom are flipped on the plots, so that 0 index in the first argument corresponds to the top.
        if self.btype == 4: #pull on bottom only
            #This puts the corect boundary values for Q on the RHS
            #Left edge
            #zero force on left
            self.FMT[1,:,0] = self.m0xx + 0.5
            self.FM1[1,:,0] = 0.5*(self.m0xx - 0.5)
            self.FM2[1,:,0] = self.m0xy
            #Right edge
            #zero force on right
            self.FMT[1,:,(self.Ngrid-1)] = self.m0xx + 0.5
            self.FM1[1,:,(self.Ngrid-1)] = 0.5*(self.m0xx - 0.5)
            self.FM2[1,:,(self.Ngrid-1)] = self.m0xy
            #Bottom edge
            self.FMT[1,0,1:(self.Ngrid-1)] = self.m0xx + self.m0yy
            self.FM1[1,0,1:(self.Ngrid-1)] = 0.5*(self.m0xx - self.m0yy)
            self.FM2[1,0,1:(self.Ngrid-1)] = self.m0xy
            #Top edge
            #zero force on top
            self.FMT[1,(self.Ngrid-1),1:(self.Ngrid-1)] = self.m0xx + 0.5
            self.FM1[1,(self.Ngrid-1),1:(self.Ngrid-1)] = 0.5*(self.m0xx - 0.5)
            self.FM2[1,(self.Ngrid-1),1:(self.Ngrid-1)] = self.m0xy
            
            #Corresponding boundary values for passive pressure
            #Left edge
            #zero force on left edge
            self.FPxx[1,:,0] = self.S0xx - self.beta * (self.m0xx - self.m0)
            self.FPyy[1,:,0] = - self.beta * (0.5 - self.m0)
            self.FPxy[1,:,0] = self.S0xy - self.beta * self.m0xy
            #Right edge
            #zero force on right
            self.FPxx[1,:,(self.Ngrid-1)] = self.S0xx - self.beta * (self.m0xx - self.m0)
            self.FPyy[1,:,(self.Ngrid-1)] = - self.beta * (0.5 - self.m0)
            self.FPxy[1,:,(self.Ngrid-1)] = self.S0xy - self.beta * self.m0xy
            #Bottom edge
            self.FPxx[1,0,:] = self.S0xx - self.beta * (self.m0xx - self.m0)
            self.FPyy[1,0,:] = self.S0yy - self.beta * (self.m0yy - self.m0)
            self.FPxy[1,0,:] = self.S0xy - self.beta * self.m0xy
            #Top edge
            #zero force on top 
            self.FPxx[1,(self.Ngrid-1),:] = self.S0xx - self.beta * (self.m0xx - self.m0)
            self.FPyy[1,(self.Ngrid-1),:] =  - self.beta * (0.5 - self.m0)
            self.FPxy[1,(self.Ngrid-1),:] = self.S0xy - self.beta * self.m0xy
            
        else:
                 
            #This puts the corect boundary values for Q on the RHS
            #Left edge
            self.FMT[1,:,0] = self.m0xx + self.m0yy
            self.FM1[1,:,0] = 0.5*(self.m0xx - self.m0yy)
            self.FM2[1,:,0] = self.m0xy
            #Right edge
            self.FMT[1,:,(self.Ngrid-1)] = self.m0xx + self.m0yy
            self.FM1[1,:,(self.Ngrid-1)] = 0.5*(self.m0xx - self.m0yy)
            self.FM2[1,:,(self.Ngrid-1)] = self.m0xy
            #Bottom edge
            self.FMT[1,0,1:(self.Ngrid-1)] = self.m0xx + self.m0yy
            self.FM1[1,0,1:(self.Ngrid-1)] = 0.5*(self.m0xx - self.m0yy)
            self.FM2[1,0,1:(self.Ngrid-1)] = self.m0xy
            #Top edge
            self.FMT[1,(self.Ngrid-1),1:(self.Ngrid-1)] = self.m0xx + self.m0yy
            self.FM1[1,(self.Ngrid-1),1:(self.Ngrid-1)] = 0.5*(self.m0xx - self.m0yy)
            self.FM2[1,(self.Ngrid-1),1:(self.Ngrid-1)] = self.m0xy
            
            #Corresponding boundary values for passive pressure
            #Left edge
            self.FPxx[1,:,0] = self.S0xx - self.beta * (self.m0xx - self.m0)
            self.FPyy[1,:,0] = self.S0yy - self.beta * (self.m0yy - self.m0)
            self.FPxy[1,:,0] = self.S0xy - self.beta * self.m0xy
            #Right edge
            self.FPxx[1,:,(self.Ngrid-1)] = self.S0xx - self.beta * (self.m0xx - self.m0)
            self.FPyy[1,:,(self.Ngrid-1)] = self.S0yy - self.beta * (self.m0yy - self.m0)
            self.FPxy[1,:,(self.Ngrid-1)] = self.S0xy - self.beta * self.m0xy
            #Bottom edge
            self.FPxx[1,0,:] = self.S0xx - self.beta * (self.m0xx - self.m0)
            self.FPyy[1,0,:] = self.S0yy - self.beta * (self.m0yy - self.m0)
            self.FPxy[1,0,:] = self.S0xy - self.beta * self.m0xy
            #Top edge
            self.FPxx[1,(self.Ngrid-1),:] = self.S0xx - self.beta * (self.m0xx - self.m0)
            self.FPyy[1,(self.Ngrid-1),:] = self.S0yy - self.beta * (self.m0yy - self.m0)
            self.FPxy[1,(self.Ngrid-1),:] = self.S0xy - self.beta * self.m0xy
        
        #----------------------------------------------------------------- 
        
    def update_current_values(self):
        #Moving on to next time step
        self.FPxx[0,:,:] = self.FPxx[1,:,:]
        self.FPyy[0,:,:] = self.FPyy[1,:,:]
        self.FPxy[0,:,:] = self.FPxy[1,:,:]
        self.FMT[0,:,:] = self.FMT[1,:,:]
        self.FM1[0,:,:] = self.FM1[1,:,:]
        self.FM2[0,:,:] = self.FM2[1,:,:]                  
    
    #SAVING-----------------------------------------------------------
    
    def save_params(self, dirname, nskip):
        
        params = {'sys_size': self.sys_size, 'Ngrid': self.Ngrid, 'nskip': nskip, 'S0xx': self.S0xx, 'S0yy': self.S0yy, 'k0': self.k0,
                                           'tm': self.tmyosin, 'tv': self.tviscous,
                                           'Blk': self.visc_bulk / self.tviscous,
                                           'Shr': self.visc_shear / self.tviscous, 'Frc': self.friction,
                                           'D': self.D, 'flow_align': self.flow_align, 'beta': self.beta, 'gradlim': self.gradlim}
    
        pickle.dump(params, open(dirname+'/params.p', 'wb'))
        
    def save_state(self, step_number, dirname, nskip):
            
        Mxx = 0.5*self.MT + self.M1
        Myy = 0.5*self.MT - self.M1
        Mxy = self.M2
        
        data = {'t': step_number * self.dt, 'Mxx': Mxx,
                                'Myy': Myy, 'Mxy': Mxy, 'Pxx': self.Pxx, 'Pyy': self.Pyy, 'Pxy': self.Pxy, 'vx': self.vx, 'vy': self.vy,
                                'gammadotxx': self.gammadotxx, 'gammadotyy': self.gammadotyy, 'gammadotxy': self.gammadotxy, 'where_pressure_diffusion': self.where_pressure_diffusion}
            
        # pickle.dump(data, open(dirname+f'/data_{int(step_number/nskip)}.pickle', 'wb'))
        pickle.dump(data, open(dirname+'/data_{}.pickle'.format(int(step_number/nskip)), 'wb'))
                
# # Implementing the "collisions" NASA EVOLVE model
# 
# Kerry N. Wood (kerry.wood@jhaupl.edu)
# 
# June 20, 2022
# 
# ~~~
# Johnson, Nicholas L., Paula H. Krisko, J-C. Liou, and Phillip D. Anz-Meador. "NASA's new breakup model of EVOLVE 4.0." Advances in Space Research 28, no. 9 (2001): 1377-1384.
# ~~~
# 
# https://www.sciencedirect.com/science/article/abs/pii/S0273117701004239


import numpy as np
import sys
import pandas as pd  # for output

# -----------------------------------------------------------------------------------------------------
# these functions help with distribution calcs
# lamc : np.log10( charlen )
# -----------------------------------------------------------------------------------------------------
def alpha_sc( lamc ):
    if lamc < -1.95: return 0
    if -1.95 < lamc and lamc < 0.55: return 0.3 + 0.4 * ( lamc + 1.2 )
    return 1
    
def mu1_sc( lamc ):
    if lamc <= -1.1: return -0.6
    if -1.1 < lamc  and lamc < 0: return -0.6 - 0.318 * (lamc + 1.1)
    return -0.95

def sigma1_sc( lamc ):
    if lamc <= -1.3: return 0.1
    if -1.3 < lamc and lamc < -0.3: return 0.1 + 0.2 * (lamc + 1.3)
    return 0.3

def mu2_sc( lamc ):
    if lamc <= -0.7: return -1.2
    if -0.7 < lamc and lamc < -0.1: return -1.2 - 1.333 * (lamc + 0.7)
    return -2.0

def sigma2_sc( lamc ):
    if lamc <= -0.5: return 0.5
    if -0.5 < lamc and lamc < -0.3: return 0.5 - (lamc + 0.5)
    return 0.3

#############################################################
def AM_gt_11( lamc  ):
    norm1 = np.random.normal( mu1_sc(lamc), sigma1_sc(lamc) )
    norm2 = np.random.normal( mu2_sc(lamc), sigma2_sc(lamc) )
    alpha = alpha_sc( lamc )
    inlog = alpha * norm1 + ( 1 - alpha ) * norm2
    return 10. ** inlog

def mu_soc( lamc ):
    if lamc <= -1.75: return -0.3
    if -1.75 < lamc and lamc < -1.25: return -0.3 - 1.4 * (lamc + 1.75)
    return -1.0

def sigma_soc( lamc ):
    if lamc <= -3.5: return 0.2
    return 0.2 + 0.1333 * (lamc + 3.5)

#############################################################
def AM_lt_08( lamc  ):
    norm = np.random.uniform( mu_soc(lamc), sigma_soc(lamc) )
    return 10.0 ** norm

#############################################################
def AM_08_11( lamc ):
    # interplate between the two boundaries; there's some question whether the bounds should be set to
    # 0.08 and 0.11, or should use the input lamc
    charlen = 10 ** lamc  # need charlen to do the interpolation...
    P1      = AM_gt_11( lamc ) #0.08 )
    P0      = AM_lt_08( lamc ) #0.11 )
    return P0 + (charlen-0.08) * (P1-P0) / 0.03
    
#############################################################
def AM( charlen ):
    lamc = np.log10( charlen )
    if charlen > 0.11: return AM_gt_11( lamc )
    if charlen < 0.08 : return AM_lt_08( lamc )
    return AM_08_11( lamc )

#############################################################
def find_area( charlen ):
    if charlen < 0.00167: return 0.540424 * charlen ** 2
    return 0.556945 * charlen ** 2.0047077
    
###########################################################################################
class evolve_collision:
    def __init__( self, 
                 mass1, mass2,      # mass of the objects
                 vel1, vel2,        # velocities (in inertial, km/s) 
                 charlen1, charlen2,# size of the objects (characteristic length)
                 min_characteristic_length = 0.01 ): 
        # we need to know which object is larger....
        vel1 = np.array(vel1)
        vel2 = np.array(vel2)
        if mass1 >= mass2 :
            self.mass1    = mass1 # kg
            self.mass2    = mass2 # kg
            self.vel1     = vel1  # vector (km/s at time of collision)
            self.vel2     = vel2  # vector
            self.charlen1 = charlen1
            self.charlen2 = charlen2
        else: # make sure mass1 is the bigger one
            self.mass1    = mass2 # kg
            self.mass2    = mass1 # kg
            self.vel1     = vel2  # vector (km/s at time of collision)
            self.vel2     = vel1  # vector
            self.charlen1 = charlen2
            self.charlen2 = charlen1
        # collision velocity and magnitude
        self.c_vel        = self.vel2 - self.vel1
        self.c_vel_mag    = np.linalg.norm( self.c_vel )
        self.relative_ke  = 0.5 * self.mass2 * ((1e3*self.c_vel_mag)**2)  # relative KE of smaller to larger object
        self.minchar      = min_characteristic_length
        self.input_mass   = self.mass1 + self.mass2 
        self.doRun()
        
    def doRun( self ):
        self.calc_collision_mass()
        self.generate_fragment_count()
        self.maxchar = np.max( (self.charlen1, self.charlen2 ) )
        self.generate_charlens()
        self.generate_AM_ratios()
        self.generate_areas()
        self.mass = self.areas / self.am_ratio
        self.generate_velocities()
            
    def calc_collision_mass( self ):
        # if the relative kinetic energy of the smaller object divided by the mass of the larger object is 
        # >= 40J/g, the collision is catastrohpic
        # if catastrophic : mass is sum of two masses 
        # if not catastrophic: mass is the smaller mass multiplied by collision velocity (km/s)
        # pg 1379 : under "Collisions"
        if (self.relative_ke / (1e3 *self.mass1 ) ) >= 40: # in GRAMS
            self.catastrophic   = True
            self.collision_mass = self.mass1 + self.mass2
        else: 
            self.catastrophic   = False
            self.collision_mass = self.mass2 * self.c_vel_mag  # kg and km/s respectively
    
    def generate_fragment_count( self ):
        # Equation 4, page 1379
        self.fragment_count = int(  0.1 * pow(self.collision_mass, 0.75) * pow(self.minchar, -1.71)  )
        return self.fragment_count
    
    # def generate_charlens( self, EXP=-1.71 ):
        # generate samples from the power law that describes fragment count....
        # the power law is from Equation 4
        # sampling a power law
        # https://mathworld.wolfram.com/RandomNumber.html
        # https://stats.stackexchange.com/questions/310610/generating-random-samples-from-a-power-law-and-testing-them-with-r-igraph
        # rands = np.random.uniform( size=self.fragment_count )
        # tval  = self.maxchar**EXP - self.minchar**EXP * rands + self.minchar**EXP
        # self.charlens =  tval**(1/EXP)

    def generate_charlens( self, EXP=-1.71 ):
        # x = [(x1^(n+1) - x0^(n+1))*y + x0^(n+1)]^(1/(n+1))
        # where y is a uniform variate, n is the distribution power, 
        # x0 and x1 define the range of the distribution, and x is your power-law distributed variate.
        rands = np.random.uniform( size=self.fragment_count )
        tv    = (self.maxchar ** (1+EXP) - self.minchar ** (1+EXP)) * rands + self.minchar ** (1 + EXP) 
        tv    = tv ** ( 1 / (EXP+1) )
        self.charlens = tv
        
    def generate_AM_ratios( self ):
        self.am_ratio = np.zeros( len(self.charlens) )
        self.am_ratio = np.array([AM(X) for X in self.charlens])
        
    def generate_areas( self ):
        self.areas = np.array( [find_area(C) for C in self.charlens])
        
    def generate_velocities( self ):
        # Eq 12, page 1383 (no units are provided here, m/s?)
        chi      = np.log10( self.am_ratio )
        mu_coll  = 0.9 * chi + 2.9
        std_coll = 0.4
        self.dv  = np.random.uniform( mu_coll, std_coll )
        self.dv  = 10 ** self.dv
        
    def match_mass( self ):
        # if mass is badly over-represented in the output values
        oldlen = len(self.charlens)
        oldmass = np.sum( self.mass )
        while np.sum( self.mass ) > self.input_mass :
            #  find the piece we can remove to get closest to the mass limit
            del_mass      = np.sum( self.mass ) - self.input_mass
            idx           = np.argmin( np.abs( del_mass - self.mass ) )
            # delete it
            self.charlens = np.delete( self.charlens, idx )
            self.mass     = np.delete( self.mass, idx )
            self.am_ratio = np.delete( self.am_ratio, idx )
            self.dv       = np.delete( self.dv, idx )
            self.areas    = np.delete( self.areas, idx )
        self.fragment_count = len(self.charlens)
        sys.stderr.write('Removed {} objects to enforce mass limit (collision_mass: {:8.3f} oldmass: {:8.3f} / newmass: {:8.3f})\n'.format( 
            oldlen - self.fragment_count, self.collision_mass,oldmass, np.sum(self.mass) ))
        sys.stderr.flush()

    def sim_output( self ):
        self.match_mass()  # drop particles until we are near "collision_mass"
        vels        = [ self.vel1, self.vel2 ]  # assingn delta-V randomly to the particles input vectors
                                                # KNW: TODO: this should correlate with impactor mass (fix later)
        idx         = np.random.choice( [0,1], size=self.fragment_count )  # pick which input velocity
        rand_sphere = np.random.normal( size=(self.fragment_count,3) )    # generate uniform random over sphere
        rand_mag    = np.linalg.norm( rand_sphere, axis=1 )
        rand_vec    = rand_sphere / rand_mag[:,np.newaxis]
        del_V       = self.dv[:,np.newaxis] * rand_vec
        to_return   = pd.DataFrame()
        to_return.index.name  = 'particle'
        to_return['mass']     = self.mass 
        to_return['A/M']      = self.am_ratio
        to_return['area']     =  self.areas 
        to_return['vel_x']    = del_V[:,0]
        to_return['vel_y']    = del_V[:,1]
        to_return['vel_z']    = del_V[:,2]
        return to_return



                
# =====================================================================================================
if __name__ == "__main__":
    # 1000 kg and 100 kg object colliding at ~14km/s
    S = evolve_collision( 1000, 100,        # masses
                      [7,0,0], [-7,0,0],    # velocity vecs
                      10, 1,                # charlens 
                      min_characteristic_length=0.05 )
    #print('Charlens   : {}'.format( S.charlens ))
    #print("Masses     : {}".format( S.mass ) )
    #print('Input  mass: {}'.format( S.collision_mass ) )
    #print('Output mass: {:8.3f}'.format( np.sum(S.mass) ) )
    #print('A/M ratios : {}'.format( S.am_ratio )) 
    print()
    print( S.sim_output() )
    
    # -----------------------------------------------------------------------------------------------------
    # 1000kg and 100kg in a NON-catastrophic collision
    S = evolve_collision( 1000, 100,        # masses
                      [7,0,0], [6.9,0,0],    # velocity vecs
                      10, 1,                # charlens 
                      min_characteristic_length=0.05 )
    #print('Charlens   : {}'.format( S.charlens ))
    #print("Masses     : {}".format( S.mass ) )
    #print('Input  mass: {}'.format( S.collision_mass ) )
    #print('Output mass: {:8.3f}'.format( np.sum(S.mass) ) )
    #print('A/M ratios : {}'.format( S.am_ratio )) 
    print()
    print( S.sim_output() )

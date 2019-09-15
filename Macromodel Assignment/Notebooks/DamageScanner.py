
import rasterio
import numpy
import pandas 
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
from rasterio.plot import show

def plot_landuse(landuse_ras,color_dict={},save=False,**kwargs):
    """
    Arguments:

        *landuse_map* : path to GeoTiff with land-use information per grid cell. 
    Optional Arguments:
        *color_dict* : Supply a dictionary with the land-use classes as keys 
        and color hex codes as values. If empty, it will use a default list (which will probably fail)
        
        *save* : Set to True if you would like to save the output. Requires 
        several **kwargs**
        
    kwargs:
        *output_path* : Specify where files should be saved.
        
        *scenario_name*: Give a unique name for the files that are going to be saved.
    """
    fig, ax = plt.subplots(1, 1,figsize=(12,10))

    if len(color_dict) == 0:
        color_dict = {
        110 : '#fb897e' ,     111 : '#b40e3e' ,     112 : '#ee0000' ,     120 : '#ee0000' , 
        130 : '#edc3c3' ,     131 : '#d97489' ,     133 : '#da6c99' ,     134 : '#da6c99' , 
        135 : '#da6c99' ,     136 : '#da6c99' ,     140 : '#331e36' ,     141 : '#363b74' , 
        142 : '#363b74' ,     143 : '#363b74' ,     144 : '#363b74' ,     150 : '#69868A' , 
        151 : '#363b74' ,     152 : '#363b74' ,     210 : '#fcfcb9' ,     211 : '#fcfcb9' , 
        220 : '#B59E99' ,     221 : '#7b323d' ,     230 : '#c3eead' ,     240 : '#b29600' , 
        250 : '#b29600' ,     310 : '#89ec46' ,     320 : '#89ec46' ,     330 : '#89ec46' , 
        340 : '#89ec46' ,     350 : '#89ec46' ,     360 : '#1c7426' ,     370 : '#e7d7bf' , 
        380 : '#3bbbb3' ,     410 : '#010000' ,     430 : '#71818e' ,     440 : '#c0c0c0' , 
        450 : '#c0c0c0' ,     460 : '#c0c0c0' ,     520 : '#545454' ,     530 : '#c39797' , 
        550 : '#7b323d' ,     560 : '#c0c0c0' ,    630 : '#b0e0e6' ,      640 : '#b0e0e6' , 
        911 : '#b0e0e6' ,    -9999: '#ffffff'
        }

    with rasterio.open(landuse_ras) as src:
        landuse = src.read()[0,:,:]
        transform = src.transform

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_axis_off()
        color_scheme_map = list(color_dict.values())

        cmap = LinearSegmentedColormap.from_list(name='landuse',
                                             colors=color_scheme_map)  

        map_dict = dict(zip(color_dict.keys(),[x for x in range(len(color_dict))]))
        # Function to be vectorized
        def map_func(val, dictionary):
            return dictionary[val] if val in dictionary else val 

        # Vectorize map_func
        vfunc  = numpy.vectorize(map_func)

        # Run
        landuse = vfunc(landuse, map_dict)

        if 'background' in kwargs:
            show(landuse,ax=ax,cmap=cmap,transform=transform,alpha=0.5)
        else:
            show(landuse,ax=ax,cmap=cmap,transform=transform)
 
    if save:
        if 'output_path' in kwargs:
            output_path = kwargs['output_path']
            if not os.path.exists(output_path):
                os.mkdir(output_path)
        if 'scenario_name' in kwargs:
            scenario_name = kwargs['scenario_name']

        fig.tight_layout()
        fig.savefig(os.path.join(output_path,'landuse_{}.png'.format(scenario_name)),dpi=350, bbox_inches='tight')


    return ax

def DamageScanner(landuse_map,inun_map,curve_path,maxdam_path):
    """
    Raster-based implementation of a direct damage assessment.
    
    Arguments:
        *landuse_map* : GeoTiff with land-use information per grid cell. Make sure 
        the land-use categories correspond with the curves and maximum damages 
        (see below). Furthermore, the resolution and extend of the land-use map 
        has to be exactly the same as the inundation map.
     
        *inun_map* : GeoTiff with inundation depth per grid cell. Make sure 
        that the unit of the inundation map corresponds with the unit of the 
        first column of the curves file.
     
        *curve_path* : File with the stage-damage curves of the different 
        land-use classes. Can also be a pandas DataFrame or numpy Array.
     
        *maxdam_path* : File with the maximum damages per land-use class 
        (in euro/m2). Can also be a pandas DataFrame or numpy Array.
     
        
    Returns:    
     *damagebin* : Table with the land-use class numbers (1st column) and the 
     damage for that land-use class (2nd column).
     
     
    """      
        
    # load land-use map
    if isinstance(landuse_map,str):
        with rasterio.open(landuse_map) as src:
            landuse = src.read()[0,:,:]
    else:
        landuse = landuse_map.copy()
    
    
    # Load inundation map
    if isinstance(inun_map,str):
        with rasterio.open(inun_map) as src:
            inundation = src.read()[0,:,:]
    else:
        inundation = inun_map.copy()
    
    inundation = numpy.nan_to_num(inundation)        

    # set cellsize:
    if isinstance(landuse_map,str) | isinstance(inun_map,str):
        cellsize = src.res[0]*src.res[1]


    # Load curves
    if isinstance(curve_path, pandas.DataFrame):
        curves = curve_path.values   
    elif isinstance(curve_path, numpy.ndarray):
        curves = curve_path
    elif curve_path.endswith('.csv'):
        curves = pandas.read_csv(curve_path).values

    #Load maximum damages
    if isinstance(maxdam_path, pandas.DataFrame):
        maxdam = maxdam_path.values 
    elif isinstance(maxdam_path, numpy.ndarray):
        maxdam = maxdam_path
    elif maxdam_path.endswith('.csv'):
        maxdam = pandas.read_csv(maxdam_path,skiprows=1).values
        
    # Speed up calculation by only considering feasible points
    inun = inundation * (inundation>=0) + 0
    inun[inun>=curves[:,0].max()] = curves[:,0].max()
    waterdepth = inun[inun>0]
    landuse = landuse[inun>0]

    # Calculate damage per land-use class for structures
    numberofclasses = len(maxdam)
    alldamage = numpy.zeros(landuse.shape[0])
    damagebin = numpy.zeros((numberofclasses, 4,))
    for i in range(0,numberofclasses):
        n = maxdam[i,0]
        damagebin[i,0] = n
        wd = waterdepth[landuse==n]
        alpha = numpy.interp(wd,((curves[:,0])),curves[:,i+1])
        damage = alpha*(maxdam[i,1]*cellsize)
        damagebin[i,1] = sum(damage)
        damagebin[i,2] = len(wd)
        if len(wd) == 0:
            damagebin[i,3] = 0
        else:
            damagebin[i,3] = numpy.mean(wd)
        alldamage[landuse==n] = damage

    # create pandas dataframe with output
    loss_df = pandas.DataFrame(damagebin.astype(float),columns=['landuse','damages','area','avg_depth']).groupby('landuse').sum()
    
    # return output
    return loss_df.sum().values[0],loss_df
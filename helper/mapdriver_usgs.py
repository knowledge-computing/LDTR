#import cv
import cv2 as cv 
import sys
import numpy as np
from subprocess import Popen
import math
import time
import os.path
import scipy.ndimage
import scipy.misc
import pickle, socket
from PIL import Image
import xml.etree.ElementTree
from time import sleep



GOOGLE_API_KEY = "enter your google api key here (static map api)"

ORIGIN_SHIFT = 2 * math.pi * 6378137 / 2.0

img_cache = {}


def lonLatToMeters(lon, lat):
    mx = lon * ORIGIN_SHIFT / 180.0
    my = math.log(math.tan((90 + lat) * math.pi / 360.0)) / (math.pi / 180.0)
    my = my * ORIGIN_SHIFT / 180.0
    return mx, my


def metersToLonLat(mx, my):
    lon = (mx / ORIGIN_SHIFT) * 180.0
    lat = (my / ORIGIN_SHIFT) * 180.0
    lat = 180 / math.pi * (2 * math.atan(math.exp(lat * math.pi / 180.0)) - math.pi / 2.0)
    return lon, lat


def DownloadMap(lat,lon, zoom, outputname):

    filename = "staticmap?center="+("%.6f" % lat)+","+("%.6f" % lon)+"&maptype=satellite&zoom="+str(zoom)+"&scale=2&style=element:labels|visibility:off&size=640x640&key=" + GOOGLE_API_KEY
    filename_shell = "staticmap?center="+("%.6f" % lat)+","+("%.6f" % lon)+"&maptype=satellite&zoom="+str(zoom)+"&scale=2&style=element:labels%7Cvisibility:off&size=640x640&key=" + GOOGLE_API_KEY
    
    #print(filename)

    Succ = False

    print(outputname)
    retry_timeout = 10

    while Succ != True :
        Popen("gtimeout 30s wget \"https://maps.googleapis.com/maps/api/"+filename+"\"", shell = True).wait()
        Popen("timeout 30s wget \"https://maps.googleapis.com/maps/api/"+filename+"\"", shell = True).wait()
        Succ = os.path.isfile(filename_shell) 
        Popen("mv \""+filename_shell+"\" "+outputname, shell=True).wait()
        if Succ != True:
            sleep(retry_timeout)
            retry_timeout += 10
            if retry_timeout > 60:
                retry_timeout = 60

            print("Retry, timeout is ", retry_timeout)
    #img = cv.LoadImage("a.jpg", 1)

    #time.sleep(0.5)

    return Succ



def GetMapAroundALoc(lat,lon, rangeInMeter, heading, folder = "googlemap/", start_lat = 42.1634, start_lon = -71.36, resolution = 1024, padding = 128, zoom = 19):
    resolution_lat = 1.0 / (111111.0)
    resolution_lon = 1.0 / (111111.0 * math.cos(start_lat / 360.0 * (math.pi * 2)))    

    min_lat = lat - resolution_lat * rangeInMeter * math.sqrt(2)
    min_lon = lon - resolution_lon * rangeInMeter * math.sqrt(2)

    max_lat = lat + resolution_lat * rangeInMeter * math.sqrt(2)
    max_lon = lon + resolution_lon * rangeInMeter * math.sqrt(2)


    x,y = lonLatToMeters(start_lon, start_lat)

    w = 2 * math.pi * 6378137 / math.pow(2, zoom)

    lon2, lat2 = metersToLonLat(x + w, y + w)
    lon1, lat1 = metersToLonLat(x - w, y - w)

    angle_per_image_lat = lat2 - lat1
    angle_per_image_lon = lon2 - lon1

    meter_per_pixel_lat = angle_per_image_lat / resolution / resolution_lat
    meter_per_pixel_lon = angle_per_image_lon / resolution / resolution_lon


    start_lat = start_lat - angle_per_image_lat * 0.5
    start_lon = start_lon - angle_per_image_lon * 0.5

    ilat_min = int(math.floor((min_lat - start_lat) / angle_per_image_lat))
    ilon_min = int(math.floor((min_lon - start_lon) / angle_per_image_lon))

    ilat_max = int(math.floor((max_lat - start_lat) / angle_per_image_lat))
    ilon_max = int(math.floor((max_lon - start_lon) / angle_per_image_lon))

    lat_n = ilat_max - ilat_min + 1
    lon_n = ilon_max - ilon_min + 1

    #print(lat_n, lon_n)

    result_image = np.zeros((lat_n * resolution, lon_n * resolution, 3), dtype=np.uint8)

    ok = True

    for i in xrange(ilat_min, ilat_max+1):
        for j in xrange(ilon_min, ilon_max+1):
            filename = folder + "sat_"+str(j)+"_"+str(i)+".png"
            Succ = os.path.isfile(filename) 

            if Succ == False:
                Succ = DownloadMap(start_lat+ angle_per_image_lat * 0.5 + i*angle_per_image_lat, start_lon + angle_per_image_lon * 0.5 + j*angle_per_image_lon, 19, filename)


            if Succ :
                subimg = scipy.ndimage.imread(filename).astype(np.uint8)

                result_image[(ilat_max - i)*resolution:(ilat_max - i + 1)*resolution,(j-ilon_min)*resolution:(j+1 -ilon_min)*resolution] = subimg[padding:resolution+padding, padding:resolution+padding]

            else :
                ok = False
                break


    #print(ilat_min, ilon_min)

    center_lat = (lat - (ilat_min * angle_per_image_lat + start_lat))/(lat_n*angle_per_image_lat)
    center_lon = (lon - (ilon_min * angle_per_image_lon + start_lon))/(lon_n*angle_per_image_lon)

    center_ilat = int((1.0-center_lat) * resolution * lat_n)
    center_ilon = int(center_lon * resolution * lon_n)



    #result_image[center_ilat-5:center_ilat+5, center_ilon-5: center_ilon+5,0] = 255
    #result_image[center_ilat-5:center_ilat+5, center_ilon-5: center_ilon+5,1] = 0
    #result_image[center_ilat-5:center_ilat+5, center_ilon-5: center_ilon+5,2] = 0

    min_d = 100000

    if min_d > center_ilat :
    	min_d = center_ilat

    if min_d > center_ilon :
    	min_d = center_ilon

    if min_d > resolution * lat_n - center_ilat:
    	min_d = resolution * lat_n - center_ilat

    if min_d > resolution * lon_n - center_ilon:
    	min_d = resolution * lon_n - center_ilon



    result_image2 = np.zeros((min_d*2+1, min_d*2+1, 3), dtype=np.uint8)

    result_image2 = result_image[center_ilat - min_d:center_ilat + min_d, center_ilon - min_d:center_ilon + min_d,:]

    #size = np.shape(result_image2)

    #newsize0 = int(size[1] * meter_per_pixel_lon / meter_per_pixel_lat) 

    #print(size[0], newsize0)

    #scale_image = scipy.misc.imresize(result_image2, (newsize0, size[1], size[2]), mode='RGB')

    #print(np.shape(scale_image))

    #img = Image.fromarray(result_image2)

    scale_image = result_image2

    img = scipy.ndimage.interpolation.rotate(scale_image, heading)

    #print(np.shape(img))


    center = np.shape(img)[0]/2

    r = int(float(rangeInMeter)/meter_per_pixel_lat)

    result = img[center-r:center+r, center-r: center+ r,:]
    

    Image.fromarray(result).save("test.png")

    #print(center_ilon, center_ilat, min_d)
    #Image.fromarray(result_image).save("test.png")

    return result, ok

def GetMapInRectEst(min_lat,min_lon, max_lat, max_lon , folder = "googlemap/", start_lat = 42.1634, start_lon = -71.36, resolution = 1024, padding = 128, zoom = 19, scale = 2):
    resolution_lat = 1.0 / (111111.0)
    resolution_lon = 1.0 / (111111.0 * math.cos(start_lat / 360.0 * (math.pi * 2)))    

    x,y = lonLatToMeters(start_lon, start_lat)

    w = 2 * math.pi * 6378137 / math.pow(2, zoom)

    lon2, lat2 = metersToLonLat(x + w, y + w)
    lon1, lat1 = metersToLonLat(x - w, y - w)

    angle_per_image_lat = lat2 - lat1
    angle_per_image_lon = lon2 - lon1

    meter_per_pixel_lat = angle_per_image_lat / resolution / resolution_lat
    meter_per_pixel_lon = angle_per_image_lon / resolution / resolution_lon


    start_lat = start_lat - angle_per_image_lat * 0.5
    start_lon = start_lon - angle_per_image_lon * 0.5

    ilat_min = int(math.floor((min_lat - start_lat) / angle_per_image_lat))
    ilon_min = int(math.floor((min_lon - start_lon) / angle_per_image_lon))

    ilat_max = int(math.floor((max_lat - start_lat) / angle_per_image_lat))
    ilon_max = int(math.floor((max_lon - start_lon) / angle_per_image_lon))

    lat_n = ilat_max - ilat_min + 1
    lon_n = ilon_max - ilon_min + 1

    return lat_n * lon_n 

def CleanCache():
    global img_cache
    img_cache = {}

def GetMapInRect(min_lat,min_lon, max_lat, max_lon , folder = "googlemap/", start_lat = 42.1634, start_lon = -71.36, resolution = 1024, padding = 128, zoom = 19, scale = 2):
    resolution_lat = 1.0 / (111111.0)
    resolution_lon = 1.0 / (111111.0 * math.cos(start_lat / 360.0 * (math.pi * 2)))    

    x,y = lonLatToMeters(start_lon, start_lat)

    w = 2 * math.pi * 6378137 / math.pow(2, zoom)

    lon2, lat2 = metersToLonLat(x + w, y + w)
    lon1, lat1 = metersToLonLat(x - w, y - w)

    angle_per_image_lat = lat2 - lat1
    angle_per_image_lon = lon2 - lon1

    meter_per_pixel_lat = angle_per_image_lat / resolution / resolution_lat
    meter_per_pixel_lon = angle_per_image_lon / resolution / resolution_lon


    start_lat = start_lat - angle_per_image_lat * 0.5
    start_lon = start_lon - angle_per_image_lon * 0.5

    ilat_min = int(math.floor((min_lat - start_lat) / angle_per_image_lat))
    ilon_min = int(math.floor((min_lon - start_lon) / angle_per_image_lon))

    ilat_max = int(math.floor((max_lat - start_lat) / angle_per_image_lat))
    ilon_max = int(math.floor((max_lon - start_lon) / angle_per_image_lon))

    lat_n = ilat_max - ilat_min + 1
    lon_n = ilon_max - ilon_min + 1

    print(lat_n, lon_n)

    result_image = np.zeros((lat_n * resolution / scale, lon_n * resolution / scale, 3), dtype=np.uint8)

    max_lat_ind = int((1.0 - (min_lat - (ilat_min * angle_per_image_lat + start_lat))/(lat_n*angle_per_image_lat))* resolution * lat_n / scale)
    min_lon_ind = int((min_lon - (ilon_min * angle_per_image_lon + start_lon))/(lon_n*angle_per_image_lon) * resolution * lon_n / scale)

    min_lat_ind = int((1.0 - (max_lat - (ilat_min * angle_per_image_lat + start_lat))/(lat_n*angle_per_image_lat))* resolution * lat_n / scale)
    max_lon_ind = int((max_lon - (ilon_min * angle_per_image_lon + start_lon))/(lon_n*angle_per_image_lon) * resolution * lon_n / scale)

    #print(min_lat_ind,min_lon_ind,max_lat_ind,max_lon_ind)


    ok = True

    for i in xrange(ilat_min, ilat_max+1):
        for j in xrange(ilon_min, ilon_max+1):

            filename = folder + "sat_"+str(j)+"_"+str(i)+".png"
            #print(filename)

            if filename in img_cache.keys():
                Succ = True

            Succ = os.path.isfile(filename) 

            if Succ == True:
                try:
                    subimg = scipy.ndimage.imread(filename).astype(np.uint8)
                except:
                    print("image file is damaged, try to redownload it", filename)
                    Succ = False

            if Succ == False:
                Succ = DownloadMap(start_lat+ angle_per_image_lat * 0.5 + i*angle_per_image_lat, start_lon + angle_per_image_lon * 0.5 + j*angle_per_image_lon, zoom, filename)

            print("total image to be downloaded", lat_n * lon_n)
            if Succ :
                #print(filename)
                if filename in img_cache.keys():
                    subimg = img_cache[filename]
                else:
                    subimg = scipy.ndimage.imread(filename).astype(np.uint8)
                    #img_cache[filename] = subimg

                if np.shape(subimg)[2] ==4:
                    subimg = subimg[:,:,0:3]


                try:
                    result_image[(ilat_max - i)*resolution/scale:(ilat_max - i + 1)*resolution/scale,(j-ilon_min)*resolution/scale:(j+1 -ilon_min)*resolution/scale] = scipy.misc.imresize(subimg[padding:resolution+padding, padding:resolution+padding],1.0/scale, mode="RGB")
                except:
                    print(np.shape(subimg))
                    ok = False
                    break

            else :
                ok = False
                break


    #print(ilat_min, ilon_min)



    result = result_image[min_lat_ind:max_lat_ind, min_lon_ind:max_lon_ind:]

    #print(center_ilon, center_ilat, min_d)
    #Image.fromarray(result_image).save("test.png")

    return result, ok




class OSMLoader:
    def __init__(self, noUnderground = False, osmfile=None, includeServiceRoad = False):

        if osmfile == None:
            filename = '/data/weiweidu/ogr2osm/CA_Bray_railroads_2001_perfect_img_sub.osm'
        else:
            filename = osmfile
        
        roadForMotorDict = {'motorway','trunk','primary','secondary','tertiary','residential'}
        roadForMotorBlackList = {'None', 'pedestrian','footway','bridleway','steps','path','sidewalk','cycleway','proposed','construction','bus_stop','crossing','elevator','emergency_access_point','escape','give_way'}


        mapxml = xml.etree.ElementTree.parse(filename).getroot()

        nodes = mapxml.findall('node')
        ways = mapxml.findall('way')
        relations = mapxml.findall('relation')

        self.nodedict = {}
        self.waydict = {}
        self.roadlist = []
        self.roaddict = {}
        self.edge2edgeid = {}
        self.edgeid2edge = {}
        self.edgeProperty = {}
        self.edgeId = 0
        way_c = 0

        for anode in nodes:
            tmp = {}
            tmp['node'] = anode
            tmp['lat'] = float(anode.get('lat'))
            tmp['lon'] = float(anode.get('lon'))
            tmp['to'] = {}
            tmp['from'] = {}

            self.nodedict.update({anode.get('id'):tmp})


        self.buildings = []

        for away in ways:
            nds = away.findall('nd')
            highway = 'None'
            lanes = -1
            width = -1
            layer = 0

            hasLane = False
            hasWidth = False
            fromMassGIS = False


            parking = False

            oneway = 0

            isBuilding = False

            building_height = 6

            cycleway = "none"

            #if highway in roadForMotorDict: #and hasLane and hasWidth and fromMassGIS: 
            #if highway not in roadForMotorBlackList:
            #if highway in roadForMotorDict:

            #if highway not in roadForMotorBlackList and parking == False:
            #if highway not in roadForMotorBlackList and (includeServiceRoad == True or parking == False): # include parking roads!
            
            idlink = []
            for anode in away.findall('nd'):
                refid = anode.get('ref')
                idlink.append(refid)

            for i in range(len(idlink)-1):
                link1 = (idlink[i], idlink[i+1])
                link2 = (idlink[i+1], idlink[i])

                if link1 not in self.edge2edgeid.keys():
                    self.edge2edgeid[link1] = self.edgeId
                    self.edgeid2edge[self.edgeId] = link1
                        #self.edgeProperty[self.edgeId] = {"width":width, "lane":lanes, "layer":layer, "roadtype": highway, "cycleway":cycleway, "info":dict(info_dict)}
                    self.edgeId += 1

                if link2 not in self.edge2edgeid.keys():
                    self.edge2edgeid[link2] = self.edgeId
                    self.edgeid2edge[self.edgeId] = link2
                        #self.edgeProperty[self.edgeId] = {"width":width, "lane":lanes, "layer":layer, "roadtype": highway, "cycleway":cycleway, "info":dict(info_dict)}
                    self.edgeId += 1


            if oneway >= 0 :
                for i in range(len(idlink)-1):
                    self.nodedict[idlink[i]]['to'][idlink[i+1]] = 1
                    self.nodedict[idlink[i+1]]['from'][idlink[i]] = 1

                self.waydict[way_c] = idlink
                way_c += 1
                    
            idlink.reverse()

            if oneway == -1:
                for i in range(len(idlink)-1):
                    self.nodedict[idlink[i]]['to'][idlink[i+1]] = 1
                    self.nodedict[idlink[i+1]]['from'][idlink[i]] = 1

                self.waydict[way_c] = idlink
                way_c += 1

            if oneway == 0:
                for i in range(len(idlink)-1):
                    self.nodedict[idlink[i]]['to'][idlink[i+1]] = 1
                    self.nodedict[idlink[i+1]]['from'][idlink[i]] = 1



if __name__ == "__main__":
    img, ok = GetMapInRect(45.49066, -122.708558, 45.509092018432014, -122.68226506517134, start_lat = 45.49066, start_lon = -122.708558, zoom=16)


    #img, ok = GetMapAroundALoc(42.279870, -71.182022, 32, 30, folder="go/googlemap/")
    #img, ok = GetMapAroundALoc(42.179870, -71.182022, 32, 30, folder="go/googlemap/")

    #result = GetClassifierResult(img)

    #print(result)
    
    #print(ok)

    #TraceQuery([42.371451, -71.066418],[42.371451 + 0.0005, -71.066418+ 0.0005], 0.0001)

    #for i in range(360):
    #    res = TraceQuery([42.371451, -71.066418],[42.371451 + 0.0005*math.cos(math.pi*2/360.0*i), -71.066418+ 0.0005*math.sin(math.pi*2/360.0*i)], 0.0001)
    #    print(res[0],res[1],res[2],res[3],res[4])




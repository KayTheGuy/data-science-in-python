import sys
import numpy as np
import pandas as pd
from math import cos, asin, sqrt
from xml.dom.minidom import parse
from xml.dom.minidom import getDOMImplementation


def output_gpx(points, output_filename):
    """ Output a GPX file with latitude and longitude from the points DataFrame """

    def append_trkpt(pt, trkseg, doc):
        trkpt = doc.createElement('trkpt')
        trkpt.setAttribute('lat', '%.8f' % (pt['lat']))
        trkpt.setAttribute('lon', '%.8f' % (pt['lon']))
        trkseg.appendChild(trkpt)

    doc = getDOMImplementation().createDocument(None, 'gpx', None)
    trk = doc.createElement('trk')
    doc.documentElement.appendChild(trk)
    trkseg = doc.createElement('trkseg')
    trk.appendChild(trkseg)

    points.apply(append_trkpt, axis=1, trkseg=trkseg, doc=doc)

    with open(output_filename, 'w') as fh:
        doc.writexml(fh, indent=' ')


def element_to_data(element):
    """ Read trkpt element and return latitude and longitude """
    latitude = float(element.getAttribute('lat'))
    longitude = float(element.getAttribute('lon'))
    return latitude, longitude


def get_data(file_name):
    """ Read GPS data from XML file """
    dom = parse(file_name)
    trkpt_elements = dom.getElementsByTagName('trkpt')  # get trkp elements
    # read lan and lon attributes in a dataframe
    gps_points = pd.DataFrame(
        list(map(element_to_data, trkpt_elements)), columns=['lat', 'lon'])
    return gps_points


def distance(gps_points):
    """ Return the distance (in metres) between the latitude/longitude points without any noise reduction """
    # get adjacent points in the same row
    gps_points['lat2'] = gps_points['lat'].shift(-1)
    gps_points['lon2'] = gps_points['lon'].shift(-1)

    def distance_helper(lat1, lon1, lat2, lon2):
        """ Return the distance between two points (Haversine formula)"""
        # taken from:
        # https://stackoverflow.com/questions/27928/calculate-distance-between-two-latitude-longitude-points-haversine-formula/21623206
        p = 0.017453292519943295 # Pi/180
        a = 0.5 - cos((lat2 - lat1) * p) / 2 + cos(lat1 * p) * \
            cos(lat2 * p) * (1 - cos((lon2 - lon1) * p)) / 2
        return 12742 * asin(sqrt(a)) * 1000 # 2*R*asin... (in meters)

    distance_helper = np.vectorize(distance_helper, otypes=[np.float])
    gps_points['dist'] = distance_helper(
        gps_points['lat'], gps_points['lon'], gps_points['lat2'], gps_points['lon2'])
    total_distance = gps_points['dist'].sum(axis=0)
    return total_distance


def smooth(data):
    """ Apply Kalman filtering on data """
    return 0


def main():
    """ Main function """
    gps_points = get_data(sys.argv[1])
    print('Unfiltered distance: %0.2f' % (distance(gps_points),))

    smoothed_points = smooth(gps_points)
    # print('Filtered distance: %0.2f' % (distance(smoothed_points),))
    # output_gpx(smoothed_points, 'out.gpx')


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 15:48:04 2020

@author: lewis
"""
__author__ = 'Lewis Picker'
__all__ = ['input_dataset']
from matplotlib import pyplot as plt
import os
from os import listdir
from os.path import isfile, join
import numpy as np
from photutils import CircularAperture, aperture_photometry
from vip_hci.fits import open_fits, write_fits
from vip_hci.preproc import frame_fix_badpix_isolated
from astropy.modeling import models, fitting
from hciplot import plot_frames
#make code shorted and within 60 charachters

class input_dataset():
    def __init__(self, inpath, outpath, dit_sci, ndit_sci,
                 ndit_sky, dit_unsat, ndit_unsat, dit_flat,
                 wavelegnth, size_telescope, pixel_scale, coro= True):
        self.inpath = inpath
        self.outpath = outpath
        #creates a list of all files in the path
        old_list = os.listdir(self.inpath)
        #removes non .fits files
        self.file_list = [file for file in  old_list if file.endswith('.fits')]
        self.dit_sci = dit_sci
        self.ndit_sci = ndit_sci
        self.ndit_sky = ndit_sky
        self.dit_unsat = dit_unsat
        self.ndit_unsat = ndit_unsat
        self.dit_flat =  dit_flat
        self.wavelength = wavelegnth
        self.size_telescope = size_telescope
        self.pixel_scale = pixel_scale
        #calculate the resoluton element.
        self.resel = (self.wavelength*180*3600)/(self.size_telescope *np.pi*
                                           self.pixel_scale)

        





#test = input_dataset('/home/lewis/Documents/Exoplanets/data_sets/HD179218/Tests/','/home/lewis/Documents/Exoplanets/data_sets/HD179218/Corrected/', 0.35, [100], [100], 0.07, [400], 0.2, 3.8e-6, 8.2, 0.02719)



    def bad_columns(self):

        bcm = np.zeros((1026, 1024) ,dtype=np.float64)
        #creating bad pixel map
        for i in range(3, 509, 8):
            for j in range(512):
                bcm[j,i] = 1


        for fname in self.file_list:
            cube_fname, header_fname = open_fits(self.inpath + fname,
                                                header = True, verbose = False)
            test_fname = cube_fname.copy()
            #crop the bad pixcel map to the same dimentions of the frames
            if len(cube_fname.shape) == 3:
                nz, ny, nx = cube_fname.shape
                cy, cx = ny/2 , nx/2
                ini_y, fin_y = int(512-cy), int(512+cy)
                ini_x, fin_x = int(512-cx), int(512+cx)
                bcm_crop = bcm[ini_y:fin_y,ini_x:fin_x]
                for j in range(nz):
                    #replace bad columns in each frame of the cubes
                    test_fname[j] = frame_fix_badpix_isolated(test_fname[j],
                                    bpm_mask= bcm_crop, sigma_clip=3,
                                    num_neig=5, size=5, protect_mask=False,
                                    radius=30, verbose=True, debug=False)
                write_fits(self.outpath + fname, test_fname,
                           header_fname, output_verify = 'fix')
            else:
                ny, nx = cube_fname.shape
                cy, cx = ny/2 , nx/2
                ini_y, fin_y = int(512-cy), int(512+cy)
                ini_x, fin_x = int(512-cx), int(512+cx)
                bcm_crop = bcm[ini_y:fin_y,ini_x:fin_x]
                test_fname = frame_fix_badpix_isolated(test_fname,
                             bpm_mask= bcm_crop, sigma_clip=3, num_neig=5,
                             size=5, protect_mask=False, radius=30,
                             verbose=True, debug=False)
                write_fits(self.outpath + fname, test_fname,
                           header_fname, output_verify = 'fix')

    def mk_dico(self, coro = True):
        if coro:
           #creacting a dictionary
           #add a list of the legnth of frames in the cubes add that to the dico
           file_list = [f for f in listdir(self.outpath) if
                        isfile(join(self.outpath, f))]
           fits_list = []
           sci_list = []
           sci_list_mjd = []
           sky_list = []
           unsat_list = []
           unsat_list_mjd = []
           flat_list = []
           X_sci_list = []
           X_unsat_list = []
           flat_dark_list = []
           sci_dark_list = []
           unsat_dark_list = []
           sci_frames = []
           sky_frames = []
           unsat_frames = []
           flat_frames = []

           for fname in file_list:
               if fname.endswith('.fits'):
                   fits_list.append(fname)
                   cube, header = open_fits(self.outpath +fname, header=True,
                                            verbose=False)
                   if header['HIERARCH ESO DPR CATG'] == 'SCIENCE'and \
                       header['HIERARCH ESO DPR TYPE'] == 'OBJECT' and \
                       header['HIERARCH ESO DET DIT'] == self.dit_sci and \
                           header['HIERARCH ESO DET NDIT'] in self.ndit_sci:

                       sci_list.append(fname)
                       sci_list_mjd.append(header['MJD-OBS'])
                       X_sci_list.append(header['AIRMASS'])
                       sci_frames.append(cube.shape[0])
                   elif (header['HIERARCH ESO DPR CATG'] == 'SCIENCE' and \
                         header['HIERARCH ESO DPR TYPE'] == 'SKY' and \
                        header['HIERARCH ESO DET DIT'] == self.dit_sci and\
                            header['HIERARCH ESO DET NDIT'] in self.ndit_sky):
                       sky_list.append(fname)
                       sky_frames.append(cube.shape[0])
                   elif header['HIERARCH ESO DPR CATG'] == 'SCIENCE' and \
                       header['HIERARCH ESO DET DIT'] == self.dit_unsat and \
                           header['HIERARCH ESO DET NDIT'] in self.ndit_unsat:
                       unsat_list.append(fname)
                       unsat_list_mjd.append(header['MJD-OBS'])
                       X_unsat_list.append(header['AIRMASS'])
                       unsat_frames.append(cube.shape[0])
                   elif 'FLAT,SKY' in header['HIERARCH ESO DPR TYPE']:
                       flat_list.append(fname)
                       flat_frames.append(cube.shape[0])
                   elif 'DARK' in header['HIERARCH ESO DPR TYPE']:
                       if header['HIERARCH ESO DET DIT'] == self.dit_flat:
                           flat_dark_list.append(fname)
                       if header['HIERARCH ESO DET DIT'] == self.dit_sci:
                           sci_dark_list.append(fname)
                       if header['HIERARCH ESO DET DIT'] == self.dit_unsat:
                           unsat_dark_list.append(fname)
           with open(self.outpath+"sci_list.txt", "w") as f:
                for sci in sci_list:
                    f.write(sci+'\n')
           with open(self.outpath+"sky_list.txt", "w") as f:
                for sci in sky_list:
                    f.write(sci+'\n')
           with open(self.outpath+"unsat_list.txt", "w") as f:
                for sci in unsat_list:
                    f.write(sci+'\n')
           with open(self.outpath+"unsat_dark_list.txt", "w") as f:
                for sci in unsat_dark_list:
                    f.write(sci+'\n')
           with open(self.outpath+"flat_dark_list.txt", "w") as f:
                for sci in flat_dark_list:
                    f.write(sci+'\n')
           with open(self.outpath+"sci_dark_list.txt", "w") as f:
                for sci in sci_dark_list:
                    f.write(sci+'\n')
           with open(self.outpath+"flat_list.txt", "w") as f:
                for sci in flat_list:
                    f.write(sci+'\n')
           
           open(self.outpath+"resel.txt", "w").write(str(self.resel))

    def find_coro_centre_list(self, file_list, coro = True, threshold = 0):

        cube = open_fits(self.outpath + file_list[0])
        nz, ny, nx = cube.shape
        #get the median frame
        median_frame = np.median(cube, axis = 0)
        shadow = np.where(median_frame >threshold, 1, 0)
        disk = models.Disk2D(amplitude = 1., x_0 = nx/2, y_0 = ny/2,
                             R_0 = nx/3)
        # Levenberg-Marquardt method for fitting
        fitter = fitting.LevMarLSQFitter()
        z,y,x = np.indices(cube.shape)
        fit = fitter(disk, x, y, shadow)
        #obtaining the fitted centre on the disk
        cy = fit.y_0.value
        cx = fit.x_0.value
        r = fit.R_0.value
        #print(cy,cx,r)
        return cy, cx, r




#The sky cubes should be identical to the science cubes without the object
#in the middle if the skylist is empty it could be caused by misclasification
#of the data (or non-existant) we can look for the sky cubes from the science
#and measure the flux at the centre of the coronagraph
#the flux at the centre of the sky cubes should be significantly lower


    def find_sky_in_sci_cube(self, nres, coro = True, debug = True):
       flux_list = []
       fname_list = []
       sci_list = []
       with open(self.outpath+"sci_list.txt", "r") as f:
            tmp = f.readlines()
            for line in tmp:
                sci_list.append(line.split('\n')[0])


       sky_list = []
       with open(self.outpath+"sky_list.txt", "r") as f:
            tmp = f.readlines()
            for line in tmp:
                sky_list.append(line.split('\n')[0])

       #draw sci list from dico
       #find corono centre of the cube.
       cy, cx, r = self.find_coro_centre_list(sci_list)
       if debug:
           plot_frames(sci_list[0][-1], circle = (cy,cx))

       #create the aperture
       circ_aper = CircularAperture((cx,cy), round(nres*self.resel))
       #total flux through the aperture
       for fname in sci_list:
           cube_fname = open_fits(self.outpath + fname)
           median_frame = np.median(cube_fname, axis = 0)
           circ_aper_phot = aperture_photometry(median_frame,
                                                    circ_aper, method='exact')
       #append it to the flux list.
           circ_flux = np.array(circ_aper_phot['aperture_sum'])
           flux_list.append(circ_flux[0])
           fname_list.append(fname)

       median_flux = np.median(flux_list)
       sd_flux = np.std(flux_list)
       for i in len(flux_list):
           if flux_list[i] < median_flux - 2*sd_flux:
               sky_list.append(fname_list[i])
               sci_list.remove(fname_list[i])
       with open(self.outpath+"sci_list.txt", "w") as f:
                for sci in sci_list:
                    f.write(sci+'\n')
       with open(self.outpath+"sky_list.txt", "w") as f:
                for sci in sky_list:
                    f.write(sci+'\n')


       #plot of centre flux
       if debug:
           plt.figure(1)
           plt.title('Flux of median sci frames at the centre \
                     of the coronagraph')
           plt.plot(flux_list, 'bo', label = 'centre flux')
           plt.ylabel('Flux')
           plt.legend()
           plt.savefig(self.outpath + 'flux_plot')
           plt.show()

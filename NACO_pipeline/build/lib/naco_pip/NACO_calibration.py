#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 16:01:17 2020

@author: lewis
"""
__author__ = 'Lewis Picker'
__all__ = ['find_agpm_list', 'raw_dataset', 'find_nearest']
import pdb
import numpy as np
import pyprind
import os
import random
from os.path import isfile
from numpy import isclose
from vip_hci.fits import open_fits, write_fits
from vip_hci.preproc import frame_crop, cube_crop_frames, frame_shift,\
    cube_subtract_sky_pca, cube_correct_nan, cube_fix_badpix_isolated
from vip_hci.var import frame_center, get_annulus_segments, frame_filter_lowpass
from vip_hci.metrics import detection
from vip_hci.conf import time_ini, timing
from hciplot import plot_frames
import naco_pip.fits_info as fits_info
#test = raw_dataset('/home/lewis/Documents/Exoplanets/data_sets/HD179218/Corrected/', '/home/lewis/Documents/Exoplanets/data_sets/HD179218/Corrected/')
#set force = true for all crops
# in the debug option for ndit plot have a plot saved showing graph 
# set the cut off value for the frames at more than 0.01% the median value  (nfr_rm) make sure to discard the first frame.
#use cube delete other options
#when checking the correction put the plots of the ndit in a debug option

#create a clas for calibration has parameters (raw_dataset(dico,coro,inpath,outpath,dit_sci,dit_unsat) )

def find_agpm_list(self, file_list, coro = True, threshold = 0):

        cube = open_fits(self.outpath + file_list[0])
        nz, ny, nx = cube.shape
        median_frame = np.median(cube, axis = 0)
        median_frame = frame_filter_lowpass(median_frame, median_size = 7, mode = 'median')
        write_fits(self.outpath +'median_frame',median_frame)        
        median_frame = frame_filter_lowpass(median_frame, mode = 'gauss',fwhm_size = 5)
        write_fits(self.outpath +'median_frame_1',median_frame)
        cy,cx = np.unravel_index(np.argmax(median_frame), median_frame.shape)
        return cy, cx
    
def find_nearest(array, value, output='index', constraint=None):
    """
    Function to find the index, and optionally the value, of an array's closest element to a certain value.
    Possible outputs: 'index','value','both'
    Possible constraints: 'ceil', 'floor', None ("ceil" will return the closest element with a value greater than 'value', "floor" the opposite)
    """
    if type(array) is np.ndarray:
        pass
    elif type(array) is list:
        array = np.array(array)
    else:
        raise ValueError("Input type for array should be np.ndarray or list.")

    idx = (np.abs(array-value)).argmin()
    if type == 'ceil' and array[idx]-value < 0:
        idx+=1
    elif type == 'floor' and value-array[idx] < 0:
        idx-=1

    if output=='index': return idx
    elif output=='value': return array[idx]
    else: return array[idx], idx

class raw_dataset():  #potentially change dico to a path to the writen list
    def __init__(self, inpath, outpath, final_sz = 0, coro = True):
        self.inpath = inpath
        self.outpath = outpath        
        self.final_sz = final_sz

        sci_list = []
        with open(self.outpath+"sci_list.txt", "r") as f:
            tmp = f.readlines()
            for line in tmp:
                sci_list.append(line.split('\n')[0])
        nx = open_fits(self.inpath + sci_list[0]).shape[2]
        self.com_sz = nx - 1
        #self.resel = 3.5154884047173294
        self.resel = float(open(self.outpath+"resel.txt", "r").readlines()[0])
        #write the text files and then draw them out by reading them
        
    def get_final_sz(self):
        
        final_sz_ori = min(2*self.agpm_pos[0]-1,2*self.agpm_pos[1]-1,2*(self.com_sz-self.agpm_pos[0])-1,2*(self.com_sz-self.agpm_pos[1])-1)
        if final_sz_ori%2 == 0:
            final_sz_ori -= 1
        final_sz = final_sz_ori
        print(final_sz)
        return final_sz



    def dark_subtract(self, verbose = True, debug = True):
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

        unsat_list = []
        with open(self.outpath+"unsat_list.txt", "r") as f:
            tmp = f.readlines()
            for line in tmp:
                unsat_list.append(line.split('\n')[0])

        unsat_dark_list = []
        with open(self.outpath+"unsat_dark_list.txt", "r") as f:
            tmp = f.readlines()
            for line in tmp:
                unsat_dark_list.append(line.split('\n')[0])

        flat_list = []
        with open(self.outpath+"flat_list.txt", "r") as f:
            tmp = f.readlines()
            for line in tmp:
                flat_list.append(line.split('\n')[0])

        flat_dark_list = []
        with open(self.outpath+"flat_dark_list.txt", "r") as f:
            tmp = f.readlines()
            for line in tmp:
                flat_dark_list.append(line.split('\n')[0])

        sci_dark_list = []
        with open(self.outpath+"sci_dark_list.txt", "r") as f:
            tmp = f.readlines()
            for line in tmp:
                sci_dark_list.append(line.split('\n')[0])
        pixel_scale = fits_info.pixel_scale


        tmp = np.zeros([3, self.com_sz, self.com_sz])
        #creating master dark cubes
        for fd, fd_name in enumerate(flat_dark_list):
            tmp_tmp = open_fits(self.inpath+fd_name, header=False, verbose=debug)
            tmp[fd] = frame_crop(tmp_tmp, self.com_sz, force = True , verbose= debug)
        write_fits(self.outpath+'flat_dark_cube.fits', tmp)
        if verbose:
            print('Flat dark cubes have been cropped and saved')

        for sd, sd_name in enumerate(sci_dark_list):
            tmp_tmp = open_fits(self.inpath+sd_name, header=False, verbose=debug)
            n_dim = tmp_tmp.ndim
            if sd == 0:
                if n_dim == 2:
                    tmp = np.array([frame_crop(tmp_tmp, self.com_sz, force = True, verbose=debug)])
                else:
                    tmp = cube_crop_frames(tmp_tmp, self.com_sz, force = True, verbose=debug)
            else:
                if n_dim == 2:
                    tmp = np.append(tmp,[frame_crop(tmp_tmp, self.com_sz, force = True, verbose=debug)],axis=0)
                else:
                    tmp = np.append(tmp,cube_crop_frames(tmp_tmp, self.com_sz, force = True, verbose=debug),axis=0)
        write_fits(self.outpath + 'sci_dark_cube.fits', tmp)
        if verbose:
            print('Sci dark cubes have been cropped and saved')

        #create an if stament for if the size is larger than sz and less than if less than crop by nx-1
        for sd, sd_name in enumerate(unsat_dark_list):
            tmp_tmp = open_fits(self.inpath+sd_name, header=False, verbose=debug)
            tmp = np.zeros([len(sci_dark_list)*tmp_tmp.shape[0],tmp_tmp.shape[1],tmp_tmp.shape[2]])
            n_dim = tmp_tmp.ndim
            if sd == 0:
                if n_dim ==2:
                    ny, nx  = tmp_tmp.shape
                    if nx <= self.com_sz:
                        tmp = np.array([frame_crop(tmp_tmp, nx - 1, force = True, verbose = debug)])
                    else:
                        tmp = np.array([frame_crop(tmp_tmp, self.com_sz, verbose = debug)])
                else:
                    nz , ny, nx = tmp_tmp.shape
                    if nx <= self.com_sz:
                        tmp = cube_crop_frames(tmp_tmp, nx-1, force = True, verbose=debug)
                    else:
                        tmp = cube_crop_frames(tmp_tmp, self.com_sz, force = True, verbose=debug)
            else:
                if n_dim == 2:
                    ny, nx = tmp_tmp.shape
                    if nx <= self.com_sz:
                        tmp = np.append(tmp,[frame_crop(tmp_tmp, nx-1, force = True, verbose=debug)],axis=0)
                    else:
                        tmp = np.append(tmp,[frame_crop(tmp_tmp, self.com_sz, force = True, verbose=debug)],axis=0)
                else:
                    nz, ny, nx = tmp_tmp.shape
                    if nx <= self.com_sz:
                        tmp = cube_crop_frames(tmp, nx - 1, force = True, verbose=debug)
                        tmp = np.append(tmp,cube_crop_frames(tmp_tmp, nx - 1, force = True, verbose=debug),axis=0)
                    else:
                        tmp = np.append(tmp,cube_crop_frames(tmp_tmp, self.com_sz, force = True, verbose=debug),axis=0)
                tmp = np.zeros([len(sci_dark_list)*tmp_tmp.shape[0],tmp_tmp.shape[1],tmp_tmp.shape[2]])
        write_fits(self.outpath+'unsat_dark_cube.fits', tmp)
        if verbose:
            print('Unsat dark cubes have been cropped and saved')


        #defining the anulus (this is where we avoid correcting around the star)
        cy, cx = find_agpm_list(self, sci_list)
        self.agpm_pos = (cx,cy)
        if verbose:
            print(' The location of the AGPM has been calculated','cy = ', cy, 'cx = ', cx)

        agpm_dedge = min(self.agpm_pos[0],self.agpm_pos[1],self.com_sz-self.agpm_pos[0],self.com_sz-self.agpm_pos[1])
        mask_arr = np.ones([self.com_sz,self.com_sz])
        cy, cx = frame_center(mask_arr)
        mask_inner_rad = int(3.0/pixel_scale) # 3arcsec min to avoid star emission
        mask_width = agpm_dedge-mask_inner_rad-1
        mask_AGPM_com = get_annulus_segments(mask_arr, mask_inner_rad, mask_width, mode = 'mask')[0]
        mask_AGPM_com = frame_shift(mask_AGPM_com,self.agpm_pos[1]-cy,self.agpm_pos[0]-cx) # agpm is not centered in the frame so shift the mask
        if verbose:
            print('AGPM mask has been defined')
        if debug:
            tmp = open_fits(self.outpath + sci_list[0])
            #plot_frames(tmp[-1], circle = self.agpm_pos)
       
        
        #now begin the dark subtraction useing PCA
        npc_dark=1 #val found this value gives the best result.
        tmp_tmp = np.zeros([len(flat_list),self.com_sz,self.com_sz])
        tmp_tmp_tmp = open_fits(self.outpath+'flat_dark_cube.fits')
        for fl, flat_name in enumerate(flat_list):
            tmp = open_fits(self.inpath+flat_name, header=False, verbose=debug)
            tmp_tmp[fl] = frame_crop(tmp, self.com_sz, force = True ,verbose=debug)
        tmp_tmp = cube_subtract_sky_pca(tmp_tmp, tmp_tmp_tmp,
                                        mask_AGPM_com, ref_cube=None, ncomp=npc_dark)
        write_fits(self.outpath+'1_crop_flat_cube.fits', tmp_tmp)
        if verbose:
            print('Dark has been subtracted from Flats')
        if debug:
            #plot the median of dark cube median of cube before subtraction median after subtraction
            tmp_tmp_tmp = np.median(tmp_tmp_tmp, axis = 0) #flat_dark cube median
            tmp = tmp #flat before subtraction 
            tmp_tmp = np.median(tmp_tmp,axis = 0) #median flat after dark subtract
            plot_frames((tmp_tmp_tmp,tmp,tmp_tmp))


        tmp_tmp_tmp = open_fits(self.outpath+'sci_dark_cube.fits')
        for sc, fits_name in enumerate(sci_list):
            tmp = open_fits(self.inpath+fits_name, header=False, verbose=debug)
            tmp = cube_crop_frames(tmp, self.com_sz, force = True, verbose=debug)
            tmp_tmp = cube_subtract_sky_pca(tmp, tmp_tmp_tmp,
                                                        mask_AGPM_com, ref_cube=None, ncomp=npc_dark)
            write_fits(self.outpath+'1_crop_'+fits_name, tmp_tmp)
        if verbose:
            print('Dark has been subtracted from Sci')
        if debug:
            #plot the median of dark cube median of cube before subtraction median after subtraction
            tmp_tmp_tmp = np.median(tmp_tmp_tmp, axis = 0)
            tmp = np.median(tmp, axis = 0)
            tmp_tmp = np.median(tmp_tmp,axis = 0)
            plot_frames((tmp_tmp_tmp,tmp,tmp_tmp))

        tmp_tmp_tmp = open_fits(self.outpath+'sci_dark_cube.fits')
        for sc, fits_name in enumerate(sky_list):
            tmp = open_fits(self.inpath+fits_name, header=False, verbose=debug)
            tmp = cube_crop_frames(tmp, self.com_sz, force = True, verbose=debug)
            tmp_tmp = cube_subtract_sky_pca(tmp, tmp_tmp_tmp,
                                                        mask_AGPM_com, ref_cube=None, ncomp=npc_dark)
            write_fits(self.outpath+'1_crop_'+fits_name, tmp_tmp)
        if verbose:
            print('Dark has been subtracted from Sky')
        if debug:
            #plot the median of dark cube median of cube before subtraction median after subtraction
            tmp_tmp_tmp = np.median(tmp_tmp_tmp, axis = 0)
            tmp = np.median(tmp, axis = 0)
            tmp_tmp = np.median(tmp_tmp,axis = 0)
            plot_frames((tmp_tmp_tmp,tmp,tmp_tmp))


        tmp_tmp_tmp = open_fits(self.outpath+'master_unsat_dark.fits')
        # no need to crop the unsat frame at the same size as the sci images if they are smaller
        for un, fits_name in enumerate(unsat_list):
            tmp = open_fits(self.inpath+fits_name, header=False)
            #tmp = cube_crop_frames(tmp,nx_unsat_crop)
            if tmp.shape[2] > self.com_sz:
                nx_unsat_crop = self.com_sz
                tmp_tmp = cube_crop_frames(tmp-tmp_tmp_tmp, nx_unsat_crop, force = True, verbose=debug)
            elif tmp.shape[2]%2 == 0:
                nx_unsat_crop = tmp.shape[2]-1
                tmp_tmp = cube_crop_frames(tmp-tmp_tmp_tmp, nx_unsat_crop, force = True, verbose=debug)
            else:
                nx_unsat_crop = tmp.shape[2]
                tmp_tmp = tmp-tmp_tmp_tmp
            write_fits(self.outpath+'1_crop_unsat_'+fits_name, tmp_tmp)
        if verbose:
            print('unsat frames have been cropped')
        if debug:
            #plot the median of dark cube median of cube before subtraction median after subtraction
            tmp_tmp_tmp = np.median(tmp_tmp_tmp, axis = 0)
            tmp = np.median(tmp, axis = 0)
            tmp_tmp = np.median(tmp_tmp,axis = 0)
            plot_frames((tmp_tmp_tmp,tmp,tmp_tmp))


# make this into a method that returns the positions its ok to leave it as a class atribute
    def find_star_unsat(self, unsat_list, verbose = True, debug = False):
        self.unsat_star_pos = {}
        y_star = []
        x_star = []
        for un, fits_name in enumerate(unsat_list):
            tmp = np.median(open_fits(self.outpath+'1_crop_unsat_'+fits_name, header=False), axis=0)
            table_res = detection(tmp,fwhm = 1.2*self.resel, bkg_sigma=1, mode='lpeaks', matched_filter=False,
                  mask=True, snr_thresh=10, plot=debug, debug=debug,
                  full_output=debug, verbose=verbose)
            y_star = np.append(x_star, table_res['y'][0])
            x_star = np.append(y_star, table_res['x'][0])

        self.unsat_star_pos['y'] = y_star
        self.unsat_star_pos['x'] = x_star
        self.unsat_star_pos['fname'] = unsat_list
        if verbose:
            print('The star has been located in the unsat cubes')
        if debug:
            snr_star = table_res['px_snr']
            for un, fits_name in enumerate(unsat_list):
                tmp = np.median(open_fits(self.outpath+'1_crop_unsat_'+fits_name, header=False), axis=0)
                plot_frames(tmp, circle = (y_star[un],x_star[un]))
                #plots all unsat



    def flat_field_correction(self, verbose = True, debug_ = False, overwrite_basic = False):
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

        flat_list = []
        with open(self.outpath+"flat_list.txt", "r") as f:
            tmp = f.readlines()
            for line in tmp:
                flat_list.append(line.split('\n')[0])

        unsat_list = []
        with open(self.outpath+"unsat_list.txt", "r") as f:
            tmp = f.readlines()
            for line in tmp:
                unsat_list.append(line.split('\n')[0])

        flat_X = []
        flat_X_values = []
        tmp = open_fits(self.outpath+'1_crop_unsat_'+unsat_list[-1],header = False)
        nx_unsat_crop = tmp.shape[2]


        for fl, flat_name in enumerate(flat_list):
            tmp, header = open_fits(self.inpath+flat_list[fl], header=True, verbose=debug_)
            flat_X.append(header['AIRMASS'])
            if fl == 0:
                flat_X_values.append(header['AIRMASS'])
            else:
                #creates a list rejecting airmass values that differ more than the tolerance.
                list_occ = [isclose(header['AIRMASS'], x, atol=0.1) for x in flat_X_values]
                if True not in list_occ:
                    flat_X_values.append(header['AIRMASS'])

        print(flat_X)
        flat_X_values = np.sort(flat_X_values) # !!! VERY IMPORTANT, DO NOT COMMENT, OR IT WILL SCREW EVERYTHING UP!!!
        print(flat_X_values)
        if verbose:
            print('The airmass values have been sorted into a list')

        # There should be 15 twilight flats in total with NACO; 5 at each airmass. BUG SOMETIMES!
        flat_tmp_cube_1 = np.zeros([5,self.com_sz,self.com_sz])
        flat_tmp_cube_2 = np.zeros([5,self.com_sz,self.com_sz])
        flat_tmp_cube_3 = np.zeros([5,self.com_sz,self.com_sz])
        counter_1 = 0
        counter_2 = 0
        counter_3 = 0

        flat_cube_3X = np.zeros([3,self.com_sz,self.com_sz])

        # TAKE MEDIAN OF each group of 5 frames with SAME AIRMASS
        flat_cube = open_fits(self.outpath+'1_crop_flat_cube.fits', header=False, verbose=debug_)
        for fl, self.flat_name in enumerate(flat_list):
            if find_nearest(flat_X_values, flat_X[fl]) == 0: #could create the function find_nearest
                flat_tmp_cube_1[counter_1] = flat_cube[fl]
                counter_1 += 1
            elif find_nearest(flat_X_values, flat_X[fl]) == 1:
                flat_tmp_cube_2[counter_2] = flat_cube[fl]
                counter_2 += 1
            elif find_nearest(flat_X_values, flat_X[fl]) == 2:
                flat_tmp_cube_3[counter_3] = flat_cube[fl]
                counter_3 += 1

        flat_cube_3X[0] = np.median(flat_tmp_cube_1,axis=0)
        flat_cube_3X[1] = np.median(flat_tmp_cube_2,axis=0)
        flat_cube_3X[2] = np.median(flat_tmp_cube_3,axis=0)
        if verbose:
            print('The median flat cubes with same airmass have been created')


        med_fl = np.zeros(3)
        gains_all = np.zeros([3,self.com_sz,self.com_sz])
#the method for avoiding the bad quadrant is removed since it is fixed in the preproc
        for ii in range(3):
            med_fl[ii] = np.median(flat_cube_3X[ii])
            gains_all[ii] = flat_cube_3X[ii]/med_fl[ii]
        master_flat_frame = np.median(gains_all, axis=0)
        if nx_unsat_crop < master_flat_frame.shape[1]:
            master_flat_unsat = frame_crop(master_flat_frame,nx_unsat_crop)
        else:
            master_flat_unsat = master_flat_frame

        write_fits(self.outpath+'master_flat_field.fits', master_flat_frame)
        write_fits(self.outpath+'master_flat_field_unsat.fits', master_flat_unsat)

        #plots(master_flat_frame, master_flat_unsat)
        if verbose:
            print('master flat frames has been saved')

        master_flat_frame = open_fits(self.outpath+'master_flat_field.fits')

        if overwrite_basic or not isfile(self.outpath+'2_ff_'+sci_list[-1]):
            for sc, fits_name in enumerate(sci_list):
                tmp = open_fits(self.outpath+'1_crop_'+fits_name, verbose=debug_)
                tmp_tmp = np.zeros_like(tmp)
                for jj in range(tmp.shape[0]):
                    tmp_tmp[jj] = tmp[jj]/master_flat_frame
                write_fits(self.outpath+'2_ff_'+fits_name, tmp_tmp, verbose=debug_)
                if not debug_:
                    os.system("rm "+self.outpath+'1_crop_'+fits_name)
        if verbose:
            print('Done scaling SCI frames with respects to ff')

        if overwrite_basic or not isfile(self.outpath+'2_ff_'+sky_list[-1]):
            for sk, fits_name in enumerate(sky_list):
                tmp = open_fits(self.outpath+'1_crop_'+fits_name, verbose=debug_)
                tmp_tmp = np.zeros_like(tmp)
                for jj in range(tmp.shape[0]):
                    tmp_tmp[jj] = tmp[jj]/master_flat_frame
                write_fits(self.outpath+'2_ff_'+fits_name, tmp_tmp, verbose=debug_)
                if not debug_:
                    os.system("rm "+self.outpath+'1_crop_'+fits_name)
        if verbose:
            print('Done scaling SKY frames with respects to ff ')

        # COMPARE BEFORE AND AFTER FLAT-FIELD
        tmp = np.median(open_fits(self.outpath+'2_ff_'+sci_list[0]), axis=0)
        tmp_tmp = np.median(open_fits(self.outpath+'2_ff_'+sci_list[-1]), axis=0)
        if debug_:
            old_tmp = np.median(open_fits(self.outpath+'1_crop_'+sci_list[0]), axis=0)
            old_tmp_tmp = np.median(open_fits(self.outpath+'1_crop_'+sci_list[-1]), axis=0)
            plot_frames(old_tmp, tmp,old_tmp_tmp, tmp_tmp)
        else:
            plot_frames(tmp, tmp_tmp)

        master_flat_unsat = open_fits(self.outpath+'master_flat_field_unsat.fits')
        for un, fits_name in enumerate(unsat_list):
            tmp = open_fits(self.outpath+'1_crop_unsat_'+fits_name, verbose=debug_)
            tmp_tmp = np.zeros_like(tmp)
            for jj in range(tmp.shape[0]):
                tmp_tmp[jj] = tmp[jj]/master_flat_unsat
            write_fits(self.outpath+'2_ff_unsat_'+fits_name, tmp_tmp, verbose=debug_)
            if not debug_:
                os.system("rm "+self.outpath+'1_crop_unsat_'+fits_name)

        if verbose:
            print('Done scaling UNSAT frames with respects to ff')

        # COMPARE BEFORE AND AFTER FLAT-FIELD
        tmp = open_fits(self.outpath+'2_ff_unsat_'+unsat_list[0])[-1]
        tmp_tmp = open_fits(self.outpath+'2_ff_unsat_'+unsat_list[-1])[-1]
        if debug_:
            old_tmp = open_fits(self.outpath+'1_crop_unsat_'+unsat_list[0])[-1]
            old_tmp_tmp = open_fits(self.outpath+'1_crop_unsat_'+unsat_list[-1])[-1]
            plot_frames(old_tmp, tmp, old_tmp_tmp, tmp_tmp)
        else:
            plot_frames(tmp, tmp_tmp)


    def correct_nan(self, verbose = True, debug = False):
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

        unsat_list = []
        with open(self.outpath+"unsat_list.txt", "r") as f:
            tmp = f.readlines()
            for line in tmp:
                unsat_list.append(line.split('\n')[0])

        n_sci = len(sci_list)
        n_sky = len(sky_list)

        bar = pyprind.ProgBar(n_sci, stream=1, title='Correcting nan pixels in SCI frames')
        for sc, fits_name in enumerate(sci_list):
            tmp = open_fits(self.outpath+'2_ff_'+fits_name, verbose=debug)
            tmp_tmp = cube_correct_nan(tmp, neighbor_box=3, min_neighbors=3, verbose=debug)
            write_fits(self.outpath+'2_nan_corr_'+fits_name, tmp_tmp, verbose=debug)
            bar.update()
            if not debug:
                os.system("rm "+self.outpath+'2_ff_'+fits_name)
        if verbose:
            print('Done corecting NAN pixels in SCI frames')


        bar = pyprind.ProgBar(n_sky, stream=1, title='Correcting nan pixels in SKY frames')
        for sk, fits_name in enumerate(sky_list):
            tmp = open_fits(self.outpath+'2_ff_'+fits_name, verbose=debug)
            tmp_tmp = cube_correct_nan(tmp, neighbor_box=3, min_neighbors=3, verbose=debug)
            write_fits(self.outpath+'2_nan_corr_'+fits_name, tmp_tmp, verbose=debug)
            bar.update()
            if not debug:
                os.system("rm "+self.outpath+'2_ff_'+fits_name)
        if verbose:
            print('Done corecting NAN pixels in SKY frames')


        for un, fits_name in enumerate(unsat_list):
            tmp = open_fits(self.outpath+'2_ff_unsat_'+fits_name, verbose=debug)
            tmp_tmp = cube_correct_nan(tmp, neighbor_box=3, min_neighbors=3, verbose=debug)
            write_fits(self.outpath+'2_nan_corr_unsat_'+fits_name, tmp_tmp, verbose=debug)
            if not debug:
                os.system("rm "+self.outpath+'2_ff_unsat_'+fits_name)
        if verbose:
            print('Done correcting NAN pixels in UNSAT frames')
            
    def correct_bad_pixels(self, verbose = True, debug = False): 
        
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
                
        unsat_list = []
        with open(self.outpath+"unsat_list.txt", "r") as f:
            tmp = f.readlines()
            for line in tmp:
                unsat_list.append(line.split('\n')[0])
        
        n_sci = len(sci_list)
        ndit_sci = fits_info.ndit_sci
        n_sky = len(sky_list)
        ndit_sky = fits_info.ndit_sky
        
        tmp = open_fits(self.outpath+'1_crop_unsat_'+unsat_list[-1],header = False)
        nx_unsat_crop = tmp.shape[2]
        
        
        master_flat_frame = open_fits(self.outpath+'master_flat_field.fits')
        # Create bpix map
        bpix = np.where(np.abs(master_flat_frame-1.09)>0.41) # i.e. for QE < 0.68 and QE > 1.5
        bpix_map = np.zeros([self.com_sz,self.com_sz])
        bpix_map[bpix] = 1
        if nx_unsat_crop < bpix_map.shape[1]:
            bpix_map_unsat = frame_crop(bpix_map,nx_unsat_crop)
        else:
            bpix_map_unsat = bpix_map
        
        #number of bad pixels
        nbpix = int(np.sum(bpix_map))
        ntotpix = self.com_sz**2
    
        print("total number of bpix: ", nbpix)
        print("total number of pixels: ", ntotpix)
        print("=> {}% of bad pixels.".format(100*nbpix/ntotpix))
    
        write_fits(self.outpath+'master_bpix_map.fits', bpix_map)
        write_fits(self.outpath+'master_bpix_map_unsat.fits', bpix_map_unsat)
        plot_frames(bpix_map, bpix_map_unsat)
        
        #update final crop size
        final_sz = self.get_final_sz()
        
        #crop frames to that size
        for sc, fits_name in enumerate(sci_list):
            tmp = open_fits(self.outpath+'2_nan_corr_'+fits_name, verbose=False)
            tmp_tmp = cube_crop_frames(tmp, final_sz, xy=self.agpm_pos)
            write_fits(self.outpath+'2_crop_'+fits_name, tmp_tmp)
            if not debug:
                os.system("rm "+self.outpath+'2_nan_corr_'+fits_name)


        for sk, fits_name in enumerate(sky_list):
            tmp = open_fits(self.outpath+'2_nan_corr_'+fits_name, verbose=False)
            tmp_tmp = cube_crop_frames(tmp, final_sz, xy=self.agpm_pos)
            write_fits(self.outpath+'2_crop_'+fits_name, tmp_tmp)
            if not debug:
                os.system("rm "+self.outpath+'2_nan_corr_'+fits_name)

        if not debug:
            tmp = open_fits(self.outpath+'2_crop_'+sci_list[0])[-1]
            tmp_tmp = open_fits(self.outpath +'2_crop_'+sci_list[-1])[-1]
            plot_frames(tmp, tmp_tmp)
        else:
            # COMPARE BEFORE AND AFTER NAN_CORR + CROP
            old_tmp = open_fits(self.outpath+'2_ff_'+sci_list[0])[-1]
            old_tmp_tmp = open_fits(self.outpath+'2_ff_'+sci_list[-1])[-1]
            tmp = open_fits(self.outpath+'2_crop_'+sci_list[0])[-1]
            tmp_tmp = open_fits(self.outpath+'2_crop_'+sci_list[-1])[-1]
            plot_frames(old_tmp, tmp, old_tmp_tmp, tmp_tmp)
            
        # Crop the bpix map in a same way
        bpix_map = open_fits(self.outpath+'master_bpix_map.fits')
        bpix_map_2ndcrop = frame_crop(bpix_map,final_sz,cenxy=self.agpm_pos)
        write_fits(self.outpath+'master_bpix_map_2ndcrop.fits', bpix_map_2ndcrop)
        
        
        
        bpix_map = open_fits(self.outpath+'master_bpix_map_2ndcrop.fits')
        t0 = time_ini()
        for sc, fits_name in enumerate(sci_list):
            tmp = open_fits(self.outpath +'2_crop_'+fits_name, verbose=False)
            # first with the bp max defined from the flat field (without protecting radius)
            tmp_tmp = cube_fix_badpix_isolated(tmp, bpm_mask=bpix_map, sigma_clip=7, num_neig=5, 
                                                           size=5, protect_mask=True, 
                                                           radius=9, verbose=False,
                                                           debug=False)
            write_fits(self.outpath+'2_bpix_corr_'+fits_name, tmp_tmp, verbose=False)
            timing(t0)
            # second, residual hot pixels
            tmp_tmp, bpm = cube_fix_badpix_isolated(tmp_tmp, bpm_mask=None, sigma_clip=8, num_neig=5, 
                                                           size=5, protect_mask=True, 
                                                           radius=10, verbose=False,
                                                           debug=False, full_output=True)
            write_fits(self.outpath+'2_bpix_corr2_'+fits_name, tmp_tmp)
            write_fits(self.outpath+'2_bpix_corr2_map_'+fits_name, bpm)
            timing(t0)
            if not debug:
                os.system("rm "+self.outpath+'2_crop_'+fits_name)
                    

        bpix_map = open_fits(self.outpath+'master_bpix_map_2ndcrop.fits')
        t0 = time_ini()
        for sk, fits_name in enumerate(sky_list):
            tmp = open_fits(self.outpath+'2_crop_'+fits_name, verbose=False)
            # first with the bp max defined from the flat field (without protecting radius)
            tmp_tmp = cube_fix_badpix_isolated(tmp, bpm_mask=bpix_map, sigma_clip=7, num_neig=5, 
                                                           size=5, protect_mask=True, 
                                                           radius=9, verbose=False,
                                                           debug=False)
            write_fits(self.outpath+'2_bpix_corr_'+fits_name, tmp_tmp, verbose=False)
            timing(t0)
            # second, residual hot pixels
            tmp_tmp, bpm = cube_fix_badpix_isolated(tmp_tmp, bpm_mask=None, sigma_clip=8, num_neig=5, 
                                                           size=5, protect_mask=True, 
                                                           radius=10, verbose=False,
                                                           debug=False, full_output=True)
            write_fits(self.outpath+'2_bpix_corr2_'+fits_name, tmp_tmp)
            write_fits(self.outpath+'2_bpix_corr2_map_'+fits_name, bpm)
            timing(t0)
            if not debug:
                os.system("rm "+self.outpath +'2_crop_'+fits_name)

    
        bpix_map_unsat = open_fits(self.outpath+'master_bpix_map_unsat.fits')
        t0 = time_ini()
        for un, fits_name in enumerate(unsat_list):
            tmp = open_fits(self.outpath+'2_nan_corr_unsat_'+fits_name, verbose=False)
            # first with the bp max defined from the flat field (without protecting radius)
            tmp_tmp = cube_fix_badpix_isolated(tmp, bpm_mask=bpix_map_unsat, sigma_clip=7, num_neig=5, 
                                                           size=5, protect_mask=True, 
                                                           radius=9, verbose=False,
                                                           debug=False)
            write_fits(self.outpath+'2_bpix_corr_unsat_'+fits_name, tmp_tmp)
            timing(t0)
            # second, residual hot pixels
            tmp_tmp, bpm = cube_fix_badpix_isolated(tmp_tmp, bpm_mask=None, sigma_clip=8, num_neig=5, 
                                                           size=5, protect_mask=True, 
                                                           radius=10, verbose=False,
                                                           debug=False, full_output=True)
            write_fits(self.outpath+'2_bpix_corr2_unsat_'+fits_name, tmp_tmp)
            write_fits(self.outpath+'2_bpix_corr2_map_unsat_'+fits_name, bpm)
            timing(t0)
            if not debug:
                os.system("rm "+ self.outpath +'2_nan_corr_unsat_'+fits_name)
                
        # FIRST CREATE MASTER CUBE FOR SCI
        tmp_tmp_tmp = open_fits(self.outpath+'2_bpix_corr2_'+sci_list[0], verbose=False)
        n_y = tmp_tmp_tmp.shape[1]
        n_x = tmp_tmp_tmp.shape[2]
        tmp_tmp_tmp = np.zeros([n_sci,n_y,n_x])
        for sc, fits_name in enumerate(sci_list):
            tmp_tmp_tmp[sc] = open_fits(self.outpath+'2_bpix_corr2_'+fits_name, verbose=False)[int(random.randrange(min(ndit_sci)))]
        tmp_tmp_tmp = np.median(tmp_tmp_tmp, axis=0)
        write_fits(self.outpath+'TMP_2_master_median_SCI.fits',tmp_tmp_tmp)
        
        # THEN CREATE MASTER CUBE FOR SKY
        tmp_tmp_tmp = open_fits(self.outpath+'2_bpix_corr2_'+sky_list[0], verbose=False)
        n_y = tmp_tmp_tmp.shape[1]
        n_x = tmp_tmp_tmp.shape[2]
        tmp_tmp_tmp = np.zeros([n_sky,n_y,n_x])
        for sk, fits_name in enumerate(sky_list):
            tmp_tmp_tmp[sk] = open_fits(self.outpath+'2_bpix_corr2_'+fits_name, verbose=False)[int(random.randrange(min(ndit_sky)))]
        tmp_tmp_tmp = np.median(tmp_tmp_tmp, axis=0)
        write_fits(self.outpath+'TMP_2_master_median_SKY.fits',tmp_tmp_tmp)
        
        bpix_map_ori = open_fits(self.outpath+'master_bpix_map_2ndcrop.fits')
        bpix_map_sci_0 = open_fits(self.outpath+'2_bpix_corr2_map_'+sci_list[0])
        bpix_map_sci_1 = open_fits(self.outpath+'2_bpix_corr2_map_'+sci_list[-1])
        bpix_map_sky_0 = open_fits(self.outpath+'2_bpix_corr2_map_'+sky_list[0])
        bpix_map_sky_1 = open_fits(self.outpath+'2_bpix_corr2_map_'+sky_list[-1])
        bpix_map_unsat_0 = open_fits(self.outpath+'2_bpix_corr2_map_unsat_'+unsat_list[0])
        bpix_map_unsat_1 = open_fits(self.outpath+'2_bpix_corr2_map_unsat_'+unsat_list[-1])
        plot_frames(bpix_map_ori, bpix_map_sci_0, bpix_map_sci_1, #tmp_tmp_tmp, #bpix_tmp,
                    bpix_map_sky_0, bpix_map_sky_1,#, #tmp_tmp_tmp_tmp, #bpix_tmp_tmp, tmpSCI-tmpSKY
                    bpix_map_unsat_0, bpix_map_unsat_1#, #tmp_tmp_tmp_tmp, #bpix_tmp_tmp, tmpSCI-tmpSKY
                   )
        
        tmpSCI = open_fits(self.outpath+'TMP_2_master_median_SCI.fits')
        tmpSKY = open_fits(self.outpath+'TMP_2_master_median_SKY.fits')
        if not debug:
            tmp_tmp = open_fits(self.outpath+'2_bpix_corr2_'+sci_list[1])[-1]
            tmp_tmp2 = open_fits(self.outpath+'2_bpix_corr2_'+sci_list[-1])[-1]
            plot_frames(#tmp, tmp-tmpSKY, #tmp_tmp_tmp, #bpix_tmp,
                        tmp_tmp, tmp_tmp - tmpSKY,#, #tmp_tmp_tmp_tmp, #bpix_tmp_tmp, tmpSCI-tmpSKY
                        #tmp2, tmp2-tmpSKY, #tmp_tmp_tmp, #bpix_tmp,
                        tmp_tmp2, tmp_tmp2 - tmpSKY#, #tmp_tmp_tmp_tmp, #bpix_tmp_tmp, tmpSCI-tmpSKY
                       )
        else:
            # COMPARE BEFORE AND AFTER BPIX CORR (without sky subtr)
            tmp = open_fits(self.outpath+'2_crop_'+sci_list[-1])[-1]
            tmp_tmp = open_fits(self.outpath+'2_bpix_corr2_'+sci_list[-1])[-1]
            tmp2 = open_fits(self.outpath+'2_crop_'+sky_list[-1])[-1]
            tmp_tmp2 = open_fits(self.outpath+'2_bpix_corr2_'+sky_list[-1])[-1]
            (tmp, tmp-tmpSKY, #tmp_tmp_tmp, #bpix_tmp,
                        tmp_tmp, tmp_tmp - tmpSKY,#, #tmp_tmp_tmp_tmp, #bpix_tmp_tmp, tmpSCI-tmpSKY
                        tmp2, tmp2-tmpSKY, #tmp_tmp_tmp, #bpix_tmp,
                        tmp_tmp2, tmp_tmp2 - tmpSKY#, #tmp_tmp_tmp_tmp, #bpix_tmp_tmp, tmpSCI-tmpSKY
                       )
from src.Structure import *
from src.CT import *
import glob
import numpy
import matplotlib.pyplot as plt
from nibabel.nicom.dicomwrappers import *
import scipy.ndimage as ndi
os.chdir('../')


# User defined variables
########################################################################################################################
# folder that contains the patient folders
parent_folder = '../patient_data/val_patients/'
# location to save the images
path = '../data/resampled_val/'
# resolution that images get resampled to in vox/mm
resolution = 0.5
# window size of interest, in mm origin at center
window_in_mm = 300
# skip duplicate patients ?
skip_duplicates = False
# keep only area above lowest dwell ?
crop_dwells = True
# number of sanity check images to plot (one per patient)
num_to_plot = 1
# make folders for every patient in image and label folders ?
make_folders = True
########################################################################################################################


# init variables
window = int(window_in_mm/resolution)
ID_num = 0
patient_number = 0
patient_names = []
one_for_training = True

def get_voxel_positions_list_version(coord):

    # Create a mesh grid with CT coordinates
    # Mesh grid consists of an array of x coordinates and an array of y coordinates
    # e.g.:x=[[0.5, 1, 1.5, 2, ... , 50, 50.5, 51, 51.5],
    #       [0.5, 1, 1.5, 2, ... , 50, 50.5, 51, 51.5],
    #       (...),
    #       [0.5, 1, 1.5, 2, ... , 50, 50.5, 51, 51.5],
    #       [0.5, 1, 1.5, 2, ... , 50, 50.5, 51, 51.5]]

    # these corrections need to be manually input to correct for interpolation errors.
    if orig_spacing[1] == 1.25:
        correction = 1/3
    elif orig_spacing[1] == 0.976562:
        correction = - 1/4
    else:
        correction = 0

    x_positions = numpy.arange(coord.num_voxels[0]) * coord.spacing[0] * coord.orient[0] + coord.img_pos[0] - correction
    y_positions = numpy.arange(coord.num_voxels[1]) * coord.spacing[1] * coord.orient[1] + coord.img_pos[1] - correction

    x_positions = x_positions[int(len(x_positions) / 2 - window / 2): int(len(x_positions) / 2 + window / 2)]
    y_positions = y_positions[int(len(y_positions) / 2 - window / 2): int(len(y_positions) / 2 + window / 2)]
    position_grid = numpy.meshgrid(x_positions, y_positions)

    # flatten mesh grid (e.g. shape: (262144, 2), 262144 [x,y] coordinates)
    position_flat = numpy.array(list(zip(position_grid[0].flatten(), position_grid[1].flatten())))
    return position_flat


for folder in sorted(os.listdir(parent_folder)):
    if "CT" in str(os.listdir(str(parent_folder)+"/"+str(folder))):

        # read in the dicom files
        dicom_folder = str(parent_folder) + "/" + str(folder)
        rs_filename_list = glob.glob(dicom_folder + '/RS*')
        rp_filename_list = glob.glob(dicom_folder + '/RP*')
        assert len(rs_filename_list) == 1, \
            "There must be one RS file, with the file name starts with RS: e.g. RSxxxxxx.dicom"
        rs_filename = rs_filename_list[0]
        rp_filename = rp_filename_list[0]
        ct_dir = os.path.dirname(rs_filename)

        ct = CT({'ct_folder': ct_dir})
        rs = dicom.read_file(rs_filename)
        rp = dicom.read_file(rp_filename)

        zoom = (ct.spacing[0] / resolution, ct.spacing[1] / resolution,
                ct.spacing[2] / resolution)

        orig_spacing = ct.spacing

        # adjust parameters for new resolution
        if not zoom == (1.0, 1.0, 1.0):
            ct.coords.slice_coordinates = ndi.interpolation.zoom(ct.coords.slice_coordinates, zoom[2])
            ct.coords.spacing = np.divide(ct.coords.spacing, zoom)
            ct.coords.num_voxels = np.multiply(ct.coords.num_voxels, zoom).astype(int)

        if skip_duplicates:
            # skip duplicate patients
            print(rs.PatientName)
            if rs.PatientName in patient_names:
                print("SKIPPING DUPLICATE PATIENT \n")
                continue
            else:
                patient_names.append(str(rs.PatientName))

        app_grid_mask_list = []
        dwell_zs = []

        ROINum_list = []
        for index, item in enumerate(rp.ApplicationSetupSequence):
            for ind, obj in enumerate(item.ChannelSequence):
                for thing in obj.BrachyControlPointSequence:
                    dwell_zs.append(thing.ControlPoint3DPosition[2])
                ROINum_list.append(int(obj.ReferencedROINumber))

        z_positions = numpy.arange(
            ct.coords.num_voxels[2]) * ct.coords.spacing[2] * ct.coords.orient[2] + ct.coords.img_pos[2]

        min_dwell_z = min(dwell_zs)

        if crop_dwells:
            idx = (np.abs(z_positions - min_dwell_z)).argmin()
        else:
            idx = 0

        grid_mask = numpy.zeros((ct.coords.num_voxels[2], window, window), dtype=numpy.bool)
        # loop through all structures in structure file
        for index, item in enumerate(rs.StructureSetROISequence):
            # make sure structure is part of the treatment (e.g. no catheters that are not part of the treatment)
            if item.ROINumber in ROINum_list:
                print("[" + str(index) + "] ROI: " + str(item.ROINumber) + ": " + item.ROIName)
                # select applicators
                if item.ROIName.startswith("Applicator") or "." in item.ROIName:
                    first_point = True
                    # get ROI Name
                    roi_name_want = item.ROIName
                    # a lof of magic happens within the 'Structure' class. See Structure.py
                    roi = Structure({'roi_name': roi_name_want, 'struct_path': rs_filename})
                    # get the contour (paths)
                    slice_data = roi.get_mask_slices(ct.coords)
                    # get flattened mesh grid with coordinates
                    positions = get_voxel_positions_list_version(ct.coords)
                    # get number of voxels ( array([x, y, z]) )
                    ct_voxels = ct.coords.num_voxels
                    # creates a 3D array of zeros with the dimensions of the voxels found above
                    _grid_mask = numpy.zeros((ct_voxels[2], window, window), dtype=numpy.bool)
                    # go through every slice and draw catheter paths on voxels to create a 3D reconstruction
                    for slice_num, paths in slice_data.items():
                        if slice_num < idx:
                            continue
                        for path in paths:
                            for contour in path:
                                contour_mask = contour.contains_points(
                                    positions).reshape((int(np.sqrt(len(positions))), int(np.sqrt(len(positions)))))

                                grid_mask[slice_num] = np.logical_or(contour_mask, grid_mask[slice_num])

        grid_mask = grid_mask[idx:]

        z_values, xph, yph = grid_mask.nonzero()
        max_z_value = max(z_values)

        if make_folders:
            # make a patient folder in image and in label folder
            os.mkdir(path + 'image/' + folder)
            os.mkdir(path + 'label/' + folder)

        ct_voxels = ct.get_unscaled_grid(zoom=zoom, window=window, idx=idx)

        for num_slice in range(0, max_z_value + 10):
            ct_img = ct_voxels[num_slice]
            label_img = grid_mask[num_slice].astype(int)

            if patient_number < num_to_plot:
                if num_slice == 50:
                    fig, ax = plt.subplots()
                    ax.imshow(ct_img)
                    y, x = label_img.nonzero()
                    ax.scatter(x, y, c='red')
                    fig.set_size_inches(19, 10.5)
                    plt.show()

            if make_folders:
                np.save(path+'image/' + folder + '/' + str(ID_num), ct_img)
                np.save(path+'label/' + folder + '/' + str(ID_num), label_img)
            else:
                np.save(path + 'image/' + str(ID_num), ct_img)
                np.save(path + 'label/' + str(ID_num), label_img)

            ID_num += 1

        patient_number += 1
        print('\n' + str(patient_number) + ' patients saved successfully \n')

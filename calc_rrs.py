import configargparse
import logging
from classes import Ispeximage
import numpy as np
import matplotlib.pyplot as plt


def compute_reflectance(grey_light_level, sky_light_level, water_light_level, fresnel_factor, grey_card_reflectance):
    """Compute the Remote Sensing Reflectance for each wavelength."""
    # Water-leaving radiance
    Lw = water_light_level - (fresnel_factor * sky_light_level)
    Lw = np.maximum(Lw, 0)  # Ensure non-negative values

    # Downwelling irradiance
    Ed = (np.pi / grey_card_reflectance) * grey_light_level

    # Compute Rrs
    valid = Ed > 1e-6
    Rrs = np.where(valid, Lw / Ed, 0)

    return Rrs


def plot_spectrum(wavelengths, r, g, b, title, filename, ylabel='Intensity'):
    plt.figure(figsize=(10, 6))
    plt.plot(wavelengths, r, 'r', label='Red Channel')
    plt.plot(wavelengths, g, 'g', label='Green Channel')
    plt.plot(wavelengths, b, 'b', label='Blue Channel')
    plt.title(title)
    plt.xlabel('Wavelength (nm)')
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.close()

def run():
    """
    Main processing
    """
    args = parse_args()

    # Load data for each measurement type
    water = Ispeximage(dng_path=args.water, save_path='.', output_plots=False)
    sky = Ispeximage(dng_path=args.sky, save_path='.', output_plots=False)
    grey = Ispeximage(dng_path=args.grey, save_path='.', output_plots=False)

    wavelengths = water.stacked_Qp[0, :]
    r_channel = water.stacked_Qp[1, :]
    g_channel = water.stacked_Qp[2, :]
    b_channel = water.stacked_Qp[3, :]

    # Compute reflectance
    # rrs_r = compute_reflectance(grey_r, sky_r, water_r)
    # rrs_g = compute_reflectance(grey_g, sky_g, water_g)
    # rrs_b = compute_reflectance(grey_b, sky_b, water_b)

    # Print results
    # print("Light Levels (mean values):")
    # print(f"Grey - Red: {np.mean(grey_r):.6f}, Green: {np.mean(grey_g):.6f}, Blue: {np.mean(grey_b):.6f}")
    # print(f"Sky  - Red: {np.mean(sky_r):.6f}, Green: {np.mean(sky_g):.6f}, Blue: {np.mean(sky_b):.6f}")
    # print(f"Water- Red: {np.mean(water_r):.6f}, Green: {np.mean(water_g):.6f}, Blue: {np.mean(water_b):.6f}")

    # print("\nRemote Sensing Reflectance (mean values):")
    # print(f"Red:   {np.mean(rrs_r):.6f}")
    # print(f"Green: {np.mean(rrs_g):.6f}")
    # print(f"Blue:  {np.mean(rrs_b):.6f}")

    # # Plot results
    # plot_spectrum(wavelengths, water_r, water_g, water_b, 'Water Spectrum', 'water_spectrum.png')
    # plot_spectrum(wavelengths, rrs_r, rrs_g, rrs_b, 'Remote Sensing Reflectance (Rrs)', 'rrs_spectrum.png', ylabel='Rrs (sr^-1)')


def parse_args():
    parser = configargparse.ArgumentParser(default_config_files=['defaults.cfg'],
                                           prog="Process iSPEX images to Rrs",
                                           formatter_class=configargparse.RawDescriptionHelpFormatter,
                                           epilog=None)

    # base configuration
    parser.add_argument('--config_file',
                        required=False,
                        is_config_file=True,
                        help="Config file that can override all the following arguments")

    parser.add_argument('-w', '--water',
                        required=True,
                        help="DNG file for water observation")

    parser.add_argument('-s', '--sky',
                        required=False,
                        help="DNG file for sky observation")

    parser.add_argument('-g', '--grey',
                        required=True,
                        help="DNG file for grey card observation")

    # constants
    parser.add_argument('--greycard_profile',
                        required=False,
                        default="constant_18p",
                        help="Reflectance profile of grey card")
    
    parser.add_argument('--fresnel_factor',
                        required=False,
                        default=0.028,
                        type=float,
                        help="Fresnel surface reflectance factor")


    # Logging options
    parser.add_argument("--log_verbosity",
                        default="INFO",
                        choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"],
                        help="Logging level")

    args, _ = parser.parse_known_args()

    return args


if __name__ == '__main__':
    log = logging.getLogger()
    log.setLevel(logging.INFO)
    console_format = '%(asctime)s - %(process)d - %(name)s - %(levelname)s - %(message)s'
    console_formatter = logging.Formatter(console_format)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(console_formatter)
    log.addHandler(console_handler)

    # file_log_format = '%(asctime)s - %(process)d - %(name)s - %(levelname)s - %(message)s'
    # file_log_formatter = logging.Formatter(file_log__format)
    # file_handler = logging.FileHandler('test.log')
    # logger.addHandler(file_handler)

    run()
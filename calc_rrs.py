import os
import configargparse
import logging
from classes import Ispeximage
import numpy as np
import matplotlib.pyplot as plt



def run():
    """
    Main processing
    """
    args = parse_args()

    # Load data for each measurement type
    water = Ispeximage(dng_path=args.water, save_path='.', output_plots=False)
    sky = Ispeximage(dng_path=args.sky, save_path='.', output_plots=False)
    card = Ispeximage(dng_path=args.grey, save_path='.', output_plots=False)


    wavelengths = water.Qp_stacked_RGB[0, :]
    water_r = remove_offset(water.Qp_stacked_RGB[1, :])
    water_g = remove_offset(water.Qp_stacked_RGB[2, :])
    water_b = remove_offset(water.Qp_stacked_RGB[3, :])

    sky_r = sky.Qp_stacked_RGB[1, :]
    sky_g = sky.Qp_stacked_RGB[2, :]
    sky_b = sky.Qp_stacked_RGB[3, :]

    card_r = card.Qp_stacked_RGB[1, :]
    card_g = card.Qp_stacked_RGB[2, :]
    card_b = card.Qp_stacked_RGB[3, :]

    # grey card profile
    if args.greycard_profile == "constant_18p":
        grey_card_reflectance = 0.18

    set_label = f"{water.label}\n{sky.label}\n{card.label}"

    # Compute reflectance
    rrs_r = compute_reflectance(water_r, sky_r, card_r, args.fresnel_factor, grey_card_reflectance)
    rrs_g = compute_reflectance(water_g, sky_g, card_g, args.fresnel_factor, grey_card_reflectance)
    rrs_b = compute_reflectance(water_b, sky_b, card_b, args.fresnel_factor, grey_card_reflectance)

    # Plot results
    plot_spectrum(wavelengths, water_r, water_g, water_b,
                  title = f"Water Spectrum\n{water.label}",
                  filename=os.path.join(args.output_path, "water_spectrum.png"),
                  ylabel='Intensity [a.u.]')

    plot_spectrum(wavelengths, rrs_r, rrs_g, rrs_b,
                  title=f"Remote Sensing Reflectance (R_rs)\n{set_label}",
                  filename=os.path.join(args.output_path, "rrs_spectrum.png"),
                  ylabel='Rrs [sr^-1]')

def remove_offset(signal):
    """Remove far-red offset from signal."""
    return signal - np.mean(signal[-10:])

def compute_reflectance(grey_signal, sky_signal, water_signal, fresnel_factor, grey_card_reflectance):
    """Compute the Remote Sensing Reflectance for each wavelength."""
    # Water-leaving radiance
    Lw = water_signal - (fresnel_factor * sky_signal)
    Lw = np.maximum(Lw, 0)  # Ensure non-negative values

    # Downwelling irradiance
    Ed = (np.pi / grey_card_reflectance) * grey_signal

    # Compute Rrs
    valid = Ed > -1e-6
    Rrs = np.where(valid, Lw / Ed, 0)
    return Rrs


def plot_spectrum(wavelengths, r, g, b, title, filename, ylabel='Intensity'):
    plt.figure(figsize=(10, 6))
    plt.plot(wavelengths, r, 'r', label='Red Channel')
    plt.plot(wavelengths, g, 'g', label='Green Channel')
    plt.plot(wavelengths, b, 'b', label='Blue Channel')
    plt.title(title)
    plt.xlabel('Wavelength [nm]')
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.close()


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

    parser.add_argument('-o', '--output_path',
                        required=False,
                        default='.',
                        help="Output path for results")

    # Logging options
    parser.add_argument("--log_verbosity",
                        default="INFO",
                        choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"],
                        help="Logging level")

    args, _ = parser.parse_known_args()

    if args.output_path is not None:
        assert os.path.exists(args.output_path)

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
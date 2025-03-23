"""
BIspectrum and Power spectrum Optimization for LArge Redshift Surveys (BIPOLARS)
-------------------------------------------------
Author: Asit Dave
Date: 23-09-2024

Description:
------------
bipolars.py is a pipeline that computes the bispectrum multipoles for Euclid-like galaxy surveys.

The pipeline is divided into four main classes:
1. FKPCatalogManager: Manages Data and Random Catalogs and helps in coversion to FKPCatalog, computes the redshift distribution n(z), and converts celestial coordinates to Cartesian coordinates.
2. FourierSpaceManager: Manages Fourier space operations, computes k-space vectors and magnitudes, and performs Fourier transforms.
3. ComputeFields: Manages field computations, constructs F0(k), F2(k), F0(x), F2(x), and triangular fields for counting triangles per bin.
4. Bispectrum: Computes the bispectrum multipoles and saves results to disk.
"""
#----------------------------------------------------------------------------------------------------------------#

# Importing required libraries
from nbodykit.lab import cosmology, FKPCatalog
from nbodykit.transform import SkyToCartesian
from nbodykit import setup_logging
from nbodykit.source.catalog import FITSCatalog
from nbodykit.algorithms import RedshiftHistogram, FKPWeightFromNbar

import numpy as np
import astropy.io.fits as fits
from tqdm import tqdm
import numba as nb
from memory_profiler import profile
import pickle
from concurrent.futures import ThreadPoolExecutor
from scipy.special import legendre

import os
import gc
import itertools
import logging
from functools import lru_cache
from typing import Tuple, List, Dict, Union
from typing_extensions import Literal

setup_logging(log_level='info')

#----------------------------------------------------------------------------------------------------------------#

class BaseCatalog:
    """
    A base class for managing catalog operations, such as loading FITS catalogs 
    and retrieving catalog metadata.

    Attributes:
    -----------
    data_catalog : FITSCatalog
        Object representing the galaxy data catalog, initialized when loaded.
    random_catalog : FITSCatalog
        Object representing the randoms catalog, initialized when loaded.

    Methods:
    --------
    load_catalogs() -> Tuple[FITSCatalog, FITSCatalog]:
        Loads the data and randoms FITS catalogs, returning the catalog objects.

    get_catalog_info() -> Dict:
        Extracts header information from the data catalog and returns it as a dictionary.

    """
    def __init__(self, data_catalog_path: str, randoms_catalog_path: str) -> None:
        self.data_catalog_path = data_catalog_path
        self.randoms_catalog_path = randoms_catalog_path
        self.data_catalog = None
        self.random_catalog = None
        self.logger = logging.getLogger(self.__class__.__name__)
        
    def load_catalog(self, catalog_path: str) -> Tuple[FITSCatalog, FITSCatalog]: # type: ignore
        """
        Loads the FITS catalogs into memory.

        Attempts to initialize the FITSCatalog objects for both the data and randoms catalogs 
        using the paths provided during class instantiation. Logs any file not found errors.

        Returns:
        --------
        Tuple[FITSCatalog, FITSCatalog]:
            A tuple containing the loaded data catalog and randoms catalog objects.

        Raises:
        -------
        FileNotFoundError
            If the specified FITS file paths cannot be located.
        Exception
            For any other error encountered during catalog loading.

        """

        self.logger.info("Loading data and randoms catalogs...")

        try:
            catalog = FITSCatalog(catalog_path)
            self.logger.info("Catalogs loaded successfully.")
        except FileNotFoundError as e:
            self.logger.error(f"File not found: {e}")
            raise e
        except Exception as e:
            self.logger.error(f"An error occurred: {e}")
            raise e
        return catalog
    
    def get_catalog_info(self, data_catalog=True, randoms_catalog=False) -> Dict:
        """
        Extracts header metadata from the data or randoms FITS catalog.

        Opens the FITS file located at `data_catalog_path` or `random_catalog_path` and retrieves the header 
        from the first HDU (Header Data Unit), returning it as a dictionary. Logs any errors encountered.
        
        Parameters:
        -----------
        data_catalog : bool
            If True, extracts header information from the data catalog. Defaults to True.
        randoms_catalog : bool
            If True, extracts header information from the randoms catalog. Defaults to False.

        Returns:
        --------
        Dict:
            A dictionary containing the header information for the specified catalog.

        Raises:
        -------
        Exception
            If an error occurs during the header extraction process.

        """
        
        hdu_info_data = None
        hdu_info_randoms = None

        if self.data_catalog_path is None or self.randoms_catalog_path is None:
            self.logger.error("Catalog paths are required. Provide paths to the data and randoms catalogs.")
            raise ValueError("Catalog paths are required. Provide paths to the data and randoms catalogs.")

        if data_catalog:
            self.logger.info("Extracting header information from the data catalog...")
            with fits.open(self.data_catalog_path) as hdul:
                try:
                    hdu_info_data = repr(hdul[1].header)
                except Exception as e:
                    self.logger.error(f"An error occurred: {e}")
                    raise e
        
        if randoms_catalog:
            self.logger.info("Extracting header information from the randoms catalog...")
            with fits.open(self.randoms_catalog_path) as hdul:
                try:
                    hdu_info_randoms = repr(hdul[1].header)
                except Exception as e:
                    self.logger.error(f"An error occurred: {e}")
                    raise e
        hdu_info = {'data': hdu_info_data, 'randoms': hdu_info_randoms}
        return hdu_info

#----------------------------------------------------------------------------------------------------------------#

class SurveyProperties:
    """
    A class for computing key survey properties from the provided data and random catalogs.

    Attributes:
    -----------
    data_catalog_path : str
        Path to the data FITS catalog.
    randoms_catalog_path : str
        Path to the randoms FITS catalog.

    Methods:
    --------
    fsky() -> float:
        Calculates the fractional sky coverage (f_sky) of the survey based on the survey area 
        relative to the total sky area.
    
    alpha() -> float:
        Calculates the alpha parameter, defined as the ratio of the number of galaxies in 
        the data catalog to the number of objects in the random catalog, a key parameter for 
        survey volume calculations.

    """
    def __init__(self, data_catalog_path: str, randoms_catalog_path: str) -> None:
        self.data_catalog_path = data_catalog_path
        self.randoms_catalog_path = randoms_catalog_path
        self.logger = logging.getLogger(self.__class__.__name__)

    def fsky(self) -> float:
        """
        Calculates the fractional sky coverage (f_sky) of the survey.

        This method accesses the 'AREA' keyword from the header of the data catalog, representing 
        the observed survey area in square degrees. The fractional sky coverage is then computed 
        by dividing the survey area by the total sky area, which is calculated in square degrees.

        Returns:
        --------
        float:
            The fractional sky coverage of the survey, `f_sky`, a value between 0 and 1.

        Raises:
        -------
        FileNotFoundError:
            If the data catalog file is not found at the specified path.
        Exception:
            For any other errors encountered during file handling or area calculation.

        """

        if self.data_catalog_path is None or self.randoms_catalog_path is None:
            self.logger.error("Catalog paths are required. Provide paths to the data and randoms catalogs.")
            raise ValueError("Catalog paths are required. Provide paths to the data and randoms catalogs.")

        self.logger.info("Computing survey area...")
        try:
            with fits.open(self.data_catalog_path) as data_catalog_astropy:
                survey_area = data_catalog_astropy['catalog'].header['AREA']  # In square degrees
        except FileNotFoundError as e:
            self.logger.error(f"File not found: {e}")
            raise e
        except Exception as e:
            self.logger.error(f"An error occurred: {e}")
            raise e
        total_sky_area = 4 * np.pi * (180 / np.pi) ** 2  # In square degrees
        fsky = survey_area / total_sky_area
        assert 0 <= fsky <= 1
        return fsky

#----------------------------------------------------------------------------------------------------------------#

class FKPCatalogManager(BaseCatalog, SurveyProperties):
    """
    A class for managing FITS-based catalogs and conducting operations related to 
    the FKP methodology, including computing n(z) distributions, converting 
    coordinates to Cartesian, and creating FKP-weighted catalogs.

    This class inherits from `BaseCatalog` for catalog management and `SurveyProperties` 
    for calculating survey properties like f_sky and alpha.

    Attributes:
    -----------
    data_catalog : FITSCatalog
        Object representing the galaxy data catalog.
    random_catalog : FITSCatalog
        Object representing the randoms catalog.
    data_catalog_path : str
        Path to the data FITS catalog.
    randoms_catalog_path : str
        Path to the randoms FITS catalog.
    cosmo : Cosmology
        Cosmological model used for coordinate transformations and redshift conversions.
    FSKY : float
        Fractional sky coverage of the survey.
    ALPHA : float
        Ratio of galaxy counts in data to random catalogs.
    logger : logging.Logger
        Logger instance for the class.

    Methods:
    --------
    add_column(data_catalog: bool, data_column_name: str, data_column_vals: np.ndarray, 
               random_catalog: bool=False, random_column_name: Union[str, None] = None, 
               random_column_vals: Union[np.ndarray, None] = None) -> Tuple[FITSCatalog, FITSCatalog]:
        Adds a new column to either or both of the data and random catalogs.

    compute_nofz(REDSHIFT: str = 'REDSHIFT', WEIGHT: str = None) -> Tuple[FITSCatalog, FITSCatalog]:
        Computes the n(z) redshift distribution for both data and random catalogs, 
        optionally applying weights if provided.

    convert_to_cartesian(REDSHIFT: str = 'REDSHIFT', RA: str = 'RA', DEC: str = 'DEC') -> Tuple[FITSCatalog, FITSCatalog]:
        Converts the RA, DEC, and redshift columns to Cartesian coordinates for both data and random catalogs.

    to_FKPCatalog(P0: float = 1e4, NZ: str = 'TOTAL_NZ') -> FKPCatalog:
        Generates an FKPCatalog with FKP weighting based on the n(z) distribution and specified P0 value.

    """
    def __init__(self, data_catalog: FITSCatalog=None, randoms_catalog: FITSCatalog=None, \
                 data_catalog_path: str=None, randoms_catalog_path: str=None, \
                   cosmo: cosmology.Cosmology=cosmology.Planck15, \
                 fsky: Union[float, None]=None, alpha: Union[float, None]=None) -> None:
        
        BaseCatalog.__init__(self, data_catalog_path, randoms_catalog_path)
        SurveyProperties.__init__(self, data_catalog_path, randoms_catalog_path)
        
        self.logger = logging.getLogger(self.__class__.__name__)

        self.cosmo = cosmo

        if fsky is None:
            try:
                self.FSKY = self.fsky()
            except Exception as e:
                self.logger.error(f"An error occurred: {e}")
                raise e
        else:
            self.FSKY = fsky
        
        self.data_catalog, self.random_catalog = self._load_catalogs(data_catalog, randoms_catalog, data_catalog_path, randoms_catalog_path)

        if alpha is None:
            try:
                Nr = len(self.random_catalog)
                Ndata = len(self.data_catalog)
                self.ALPHA = Ndata / Nr
            except Exception as e:
                self.logger.error(f"An error occurred: {e}")
                raise e
        else:
            self.ALPHA = alpha
    
    def _load_catalogs(self, data_catalog, randoms_catalog, data_catalog_path, randoms_catalog_path):
        """Load data and randoms catalogs."""
        if data_catalog is not None:
            data = data_catalog
        elif data_catalog_path is not None:
            data = self.load_catalog(data_catalog_path)
        else:
            self.logger.error("Data catalog not provided. Provide a catalog or its path.")
            raise ValueError("Data catalog not provided. Provide a catalog or its path.")
        
        if randoms_catalog is not None:
            randoms = randoms_catalog
        elif randoms_catalog_path is not None:
            randoms = self.load_catalog(randoms_catalog_path)
        else:
            self.logger.error("Randoms catalog not provided. Provide a catalog or its path.")
            raise ValueError("Randoms catalog not provided. Provide a catalog or its path.")
        
        return data, randoms

    def add_column(self, data_catalog: bool, data_column_name: str, data_column_vals: np.ndarray, \
                   random_catalog: bool=False, random_column_name: Union[str, None] = None, random_column_vals: Union[np.ndarray, None] = None) -> Tuple[FITSCatalog, FITSCatalog]: # type: ignore
        """
        Adds a new column to either or both of the data catalog and the random catalog within a FITS catalog.

        Parameters:
        ----------
        data_catalog : bool
            If True, adds the specified column to the main data catalog.
        data_column_name : str
            The name of the column to be added to the data catalog.
        data_column_vals : np.ndarray
            The values for the new column in the data catalog. Must match the length of existing catalog rows.
        random_catalog : bool, optional
            If True, adds a new column to the random catalog. Defaults to False.
        random_column_name : Union[str, None], optional
            The name of the column to add to the random catalog. Required if `random_catalog` is True.
        random_column_vals : Union[np.ndarray, None], optional
            The values for the new column in the random catalog. Must match the length of existing rows in the random catalog.
            Required if `random_catalog` is True.

        Returns:
        -------
        Tuple[FITSCatalog, FITSCatalog]
            A tuple containing the modified data catalog and random catalog.

        Raises:
        ------
        AssertionError
            If `random_catalog` is True but `random_column_name` or `random_column_vals` is not provided.
        """
        if data_catalog:
            self.data_catalog[data_column_name] = data_column_vals
        if random_catalog:
            assert random_column_name is not None and random_column_vals is not None, "Random column name and values must be provided"
            self.random_catalog[random_column_name] = random_column_vals
        return self.data_catalog, self.random_catalog
        

    def compute_nofz(self, REDSHIFT='REDSHIFT', WEIGHT=None, \
                     zhist_randoms: Union[RedshiftHistogram, None]=None, \
                        hist_save_dir: str=None, hist_file_prefix: str=None) -> Tuple[FITSCatalog, FITSCatalog]: # type: ignore
        """
        Computes the n(z) redshift distribution for both data and random catalogs.

        This method estimates the n(z) redshift distribution for the survey by creating a 
        histogram of redshift values for both the data and random catalogs. It uses the 
        `RedshiftHistogram` class from the Nbodykit library to perform this calculation. 
        The resulting n(z) distributions are then interpolated using spline interpolation 
        and stored in a new column ('NZ') within each catalog. This process facilitates 
        later use in FKP weighting and other analyses.

        Parameters:
        -----------
        REDSHIFT : str, optional
            Column name for redshift values in the catalogs (default is 'REDSHIFT').
        WEIGHT : str, optional
            Column name for weights applied to the objects in the catalog (default is None).
            If provided, the weights are included in the n(z) calculation.
        zhist_randoms : Union[RedshiftHistogram, None], optional
            Precomputed n(z) histogram for the random catalog. If provided, this histogram
            will be used instead of recalculating n(z) from the random catalog.
        hist_save_dir : str, optional
            Directory to save the computed n(z) histograms as JSON files. If provided, the
            histograms will be saved to this directory.
        hist_file_prefix : str, optional
            Prefix for the n(z) histogram files. If not provided, the default prefix is 'nz'.

        Returns:
        --------
        Tuple[FITSCatalog, FITSCatalog]
            A tuple containing the modified data and random catalogs with the computed n(z) 
            distribution added as the 'NZ' column.
            Moreover, the 'TOTAL_NZ' column is added to the catalogs, which contains the n(z)
            values derived from the full/total catalog containing all the species.

        Notes:
        ------
        - If `zhist_data` or `zhist_randoms` are not provided during initialization, this 
        method will compute n(z) from the catalogs using the `RedshiftHistogram` class.
        - If either `zhist_data` or `randoms_zhist` is provided, it will be used instead 
        of recalculating n(z) from the catalogs.

        """

        if zhist_randoms is None:
            self.logger.warning("Computing n(z) distributions from the catalogs.")

        self.logger.info("Computing n(z) distributions from the catalogs to compute I22 and I33...")
        nofz_data = RedshiftHistogram(source=self.data_catalog, 
                                        fsky=self.FSKY, 
                                        cosmo=self.cosmo, 
                                        redshift=REDSHIFT, 
                                        weight=WEIGHT
                                        )
        nofz_random = RedshiftHistogram(source=self.random_catalog, 
                                        fsky=self.FSKY, 
                                        cosmo=self.cosmo, 
                                        redshift=REDSHIFT, 
                                        weight=WEIGHT
                                        )
        self.logger.info("Generated n(z) distributions successfully.")

        self.logger.info("Interpolating n(z) redshift distributions...")
        self.data_catalog['NZ'] = self.ALPHA * nofz_random.interpolate(z=self.data_catalog[REDSHIFT], ext='extrapolate')
        self.random_catalog['NZ'] = self.ALPHA * nofz_random.interpolate(z=self.random_catalog[REDSHIFT], ext='extrapolate')
        self.logger.info("n(z) interpolated succesfully.")

        if zhist_randoms is None:
            # If the provided n(z) histograms are missing, use the computed n(z) distributions
            self.logger.warning("n(z) histograms not provided. Using catalog n(z) distributions to compute FKP weights.")
            self.data_catalog['TOTAL_NZ'] = self.data_catalog['NZ']
            self.random_catalog['TOTAL_NZ'] = self.random_catalog['NZ']

            if hist_save_dir is not None:
                os.makedirs(hist_save_dir, exist_ok=True)

                if hist_file_prefix is None:
                    hist_file_prefix = "nz"

                self.logger.info(f"Saving n(z) histograms to {hist_save_dir}")
                nofz_data.save(os.path.join(hist_save_dir, hist_file_prefix + "_data") + ".json")
                nofz_random.save(os.path.join(hist_save_dir, hist_file_prefix + "_randoms") + ".json")
                self.logger.info("n(z) histograms saved successfully.")

            elif hist_save_dir is None and zhist_randoms is None:
                self.logger.warning("n(z) histograms will not be saved. Provide a directory to save the histograms.")

            
        else:
            self.logger.info("Using provided n(z) histograms...")
            imported_nofz_random = zhist_randoms

            self.logger.info("Interpolating n(z) from the given redshift distributions...")
            self.data_catalog['TOTAL_NZ'] = self.ALPHA * imported_nofz_random.interpolate(z=self.data_catalog[REDSHIFT], ext='extrapolate')
            self.random_catalog['TOTAL_NZ'] = self.ALPHA * imported_nofz_random.interpolate(z=self.random_catalog[REDSHIFT], ext='extrapolate')
            self.logger.info("n(z) interpolated successfully.")

            del imported_nofz_random

        del nofz_data, nofz_random

        self.logger.info("n(z) computed successfully.")

        return self.data_catalog, self.random_catalog
    
    def convert_to_cartesian(self, catalog: Literal['data', 'randoms', 'both']='both', REDSHIFT='REDSHIFT', RA='RA', DEC='DEC') -> None:
        """
        Converts RA, DEC, and redshift coordinates to Cartesian coordinates for both 
        the data and random catalogs.

        Parameters:
        -----------
        catalog : Literal['data', 'randoms', 'both']
            Specifies which catalog(s) to convert to Cartesian coordinates. Defaults to 'both'.
        REDSHIFT : str, optional
            Column name for redshift values in the catalogs (default is 'REDSHIFT').
        RA : str, optional
            Column name for Right Ascension values in the catalogs (default is 'RA').
        DEC : str, optional
            Column name for Declination values in the catalogs (default is 'DEC').

        Returns:
        --------
        Tuple[FITSCatalog, FITSCatalog] or FITSCatalog:
            Returns the modified catalog(s) with Cartesian coordinates added. If 
            both catalogs are processed, a tuple is returned. Otherwise, the individual 
            catalog is returned.
        """
        self.logger.info("Converting RA, DEC, and redshift to Cartesian coordinates...")
        
        if catalog in ['data', 'both']:
            self.logger.info("Processing data catalog...")
            self.data_catalog['POSITION'] = SkyToCartesian(ra=self.data_catalog[RA], dec=self.data_catalog[DEC], 
                                                        redshift=self.data_catalog[REDSHIFT], 
                                                        cosmo=self.cosmo
                                                        )
        if catalog in ['randoms', 'both']:
            self.logger.info("Processing random catalog...")
            self.random_catalog['POSITION'] = SkyToCartesian(ra=self.random_catalog[RA], dec=self.random_catalog[DEC],
                                                            redshift=self.random_catalog[REDSHIFT], 
                                                            cosmo=self.cosmo
                                                            )
        self.logger.info("Conversion to Cartesian coordinates completed.")

        # Return the modified catalog(s) based on the input
        if catalog == 'data':
            return self.data_catalog
        elif catalog == 'randoms':
            return self.random_catalog
        elif catalog == 'both':
            return self.data_catalog, self.random_catalog

    @profile
    def to_FKPCatalog(self, P0=1e4, NZ='TOTAL_NZ') -> FKPCatalog:
        """
        Creates an FKPCatalog using the data and random catalogs, with FKP weights applied.

        The FKP weighting scheme is used to compute the weights for both the data and random 
        catalogs based on the provided n(z) distribution and P0 parameter.

        Parameters:
        -----------
        P0 : float, optional
            The P0 value used in the FKP weight calculation (default is 1e4).
        NZ : str, optional
            Column name for the n(z) values in the catalogs (default is 'TOTAL_NZ').
            'TOTAL_NZ' is used to indicate the n(z) values derived from the full/total catalog containing all the species.

        Returns:
        --------
        FKPCatalog:
            The resulting FKPCatalog with FKP weights applied.
        """
        self.logger.info("Creating FKPCatalog...")
        fkp_catalog = FKPCatalog(data=self.data_catalog, randoms=self.random_catalog, nbar=NZ)
        self.logger.info("Determining FKP weights...")
        fkp_catalog['data/FKPWeight'] = FKPWeightFromNbar(P0=P0, nbar=fkp_catalog['data/' + NZ])
        fkp_catalog['randoms/FKPWeight'] = FKPWeightFromNbar(P0=P0, nbar=fkp_catalog['randoms/' + NZ])
        
        fkp_catalog.attrs.update({'alpha': self.ALPHA, 'fsky': self.FSKY, 'P0': P0})

        self.logger.info("FKPCatalog created and FKP weights determined successfully.")
        return fkp_catalog

#----------------------------------------------------------------------------------------------------------------#

class FourierSpaceManager:
    """
    Manages the setup and calculations for Fourier space operations.

    This class handles the creation of k-space grids, k-bins, and k-configurations 
    used in various Fourier analyses. It also manages the conversion of fields 
    between real space and Fourier space.

    Attributes:
    -----------
    config : dict
        Configuration dictionary which is expected to be defined as follows:
        {
            'N': int,               # Grid size for the Fourier space.
            'L': float,             # Physical size of the box in real space (in Mpc/h).
            'dk': float,            # Bin-width of k-modes.
            'kmin': float,          # Minimum k-bin center in the k-space grid.
            'kmax': float,          # Maximum k-bin center in the k-space grid.
            'poles': List[int]      # List of multipoles (e.g., 0, 2) to consider in calculations.
        }
    N : int
        Grid size for the Fourier space.
    L : float
        Physical size of the box in real space (in Mpc/h).
    dk : float
        Bin-width of k-modes.
    kmin : float
        Minimum k-bin center in the k-space grid.
    kmax : float
        Maximum k-bin center in the k-space grid.
    poles : List[int]
        List of multipoles (e.g., 0, 2) to consider in calculations.

    Methods:
    --------
    k_grid() -> Tuple[np.ndarray, np.ndarray]:
        Computes the k-space grid (k-vectors and their magnitudes).
    get_k_bin_indices(k_magnitude) -> Tuple[List[float], List[np.ndarray]]:
        Determines the binning for the k-magnitude values.
    create_k_configurations(k_bins) -> List[Tuple[int, int, int]]:
        Creates valid k-space configurations from the k-bins.
    x_hat() -> np.ndarray:
        Computes the real-space unit vectors for the grid.
    k_hat() -> np.ndarray:
        Computes the unit vectors for k-space.
    """

    def __init__(self, config: dict):
        self.config = config
        self.N = config['N']
        self.L = config['L']
        self.dk = config.get('dk', 2 * np.pi / self.L) if self.L is not None else None
        self.kmin = config.get('kmin', 0 + self.dk/2) if self.dk is not None else None
        self.k_nyquist = np.pi * self.N / self.L if self.N is not None and self.L is not None else None
        self.kmax = config.get('kmax', self.k_nyquist) if self.k_nyquist is not None else None
        self.poles = config.get('poles', [0])
        self.logger = logging.getLogger(self.__class__.__name__)

        # Validate the poles
        valid_poles = [0, 2]
        for pole in self.poles:
            if pole not in valid_poles:
                self.logger.warning('Invalid value for poles. Choose from [0, 2]. Deafulting to [0, 2]')
                self.poles = [0, 2]

        if 0 not in self.poles:
            self.poles = [0] + self.poles
            self.logger.warning("The zero pole was not included in the poles list. Therefore, adding zero to the list.")
        assert 0 in self.poles

    @profile
    def k_grid(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes the k-space grid (k-vectors and their magnitudes) for the given N and L.

        This method creates the k-space vectors and their corresponding magnitudes in 
        Fourier space. It assumes that the grid is cubic with dimensions N x N x N.

        Returns:
        --------
        Tuple[np.ndarray, np.ndarray]:
            A tuple where the first element is an array of k-space vectors and the 
            second element is an array of their magnitudes.
        """
        
        if self.N is None or self.L is None:
            return None, None
        
        self.logger.info("Computing k-space vectors and magnitudes...")
        kspace = np.indices((self.N, self.N, self.N))
        kspace[kspace > self.N // 2] -= self.N

        k_vectors = 2 * np.pi * kspace / self.L
        del kspace
        k_magnitude = np.linalg.norm(k_vectors, axis=0)
        self.logger.info("k-space vectors and magnitudes computed successfully.")

        assert self.kmax <= np.round(np.max(k_magnitude), 1), "Input k-max is greater than the maximum possible k-vector magnitude for a given N and L. This leads to empty/undefined k-bins."

        return k_vectors, k_magnitude

    @profile
    def get_k_bin_indices(self, k_magnitude) -> Tuple[List[float], List[np.ndarray]]:
        """
        Determines the binning for the k-magnitude values and finds the indices of 
        the k-vectors that fall into each bin.

        Parameters:
        -----------
        k_magnitude : np.ndarray
            Array of k-vector magnitudes.

        Returns:
        --------
        Tuple[List[float], List[np.ndarray]]:
            A tuple containing a list of bin centers (k_bins) and a list of arrays 
            where each array contains the indices of the k-vectors that fall into 
            the corresponding bin.
        """
        self.logger.info("Getting k-bin indices...")
        if self.kmin == 0:
            self.kmin += self.dk/2
            
        assert self.kmin < self.kmax
        assert self.dk > 0

        try:
            k_bin_centers = np.arange(self.kmin, self.kmax, self.dk)
        except Exception as e:
            self.logger.error(f"An error occurred: {e}")
            raise e
        k_bin_info = {}

        for k_bin_center in tqdm(k_bin_centers):
            filtered_indices = np.where((k_magnitude >= k_bin_center - self.dk / 2) &
                                        (k_magnitude < k_bin_center + self.dk / 2))
            
            bin_k_avg = np.mean(k_magnitude[filtered_indices])
            k_bin_info[bin_k_avg] = filtered_indices
        
        self.logger.info("k-bin indices computed successfully.") 
        self.logger.info(f"Number of k-bins: {len(k_bin_info.values())}")
        
        k_bin_indices = list(k_bin_info.values())
        k_bins = list(k_bin_info.keys())
        self.k_bin_info = k_bin_info
        return k_bins, k_bin_indices

    @profile
    def create_k_configurations(self, k_bins) -> List[Tuple[int, int, int]]:
        """
        Creates valid k-space configurations (k1, k2, k3) from the k-bins, such that k1 >= k2 >= k3 
        and satisfies the triangular inequality condition (k2 + k3 >= k1).

        Parameters:
        -----------
        k_bins : List[float]
            A list of k-bin centers.

        Returns:
        --------
        List[Tuple[int, int, int]]:
            A list of tuples representing valid k-configurations in Fourier space.
        """
        self.logger.info("Creating k-configurations...")
        k_configs = []
        for k1 in range(1, len(k_bins) + 1):
            for k2 in range(1, k1 + 1):
                for k3 in range(1, k2 + 1):
                    if (k2+k3 >= k1) and k1 >= k2 >= k3:
                        k_configs.append((k1, k2, k3))
        self.logger.info(f"Number of k-configurations: {len(k_configs)}")
        return np.array(k_configs)
    
    @profile
    def x_hat(self) -> np.ndarray:
        """
        Computes the Real-space (x-space) unit vectors for the real-space grid.

        Returns:
        --------
        np.ndarray:
            An array of unit vectors in Real-space corresponding to each grid point.
        """
        logging.info("Computing x-space vectors/grid...")
        xhat = np.indices((self.N, self.N, self.N), dtype=np.float64)
        xhat[xhat > self.N // 2] -= self.N
        xhat *= self.L / self.N
        xhat += self.BoxCenter[:, None, None, None]
        xhat /= np.linalg.norm(xhat, axis=0)
        self.logger.info("x-space grid computed successfully.")
        return xhat
    
    @profile
    def k_hat(self) -> np.ndarray:
        """
        Computes the unit vectors for k-space.

        Returns:
        --------
        np.ndarray:
            An array of unit vectors in k-space corresponding to each grid point.
        """
        self.logger.info("Computing k-space grid...")
        khat = self.k_vectors / (np.linalg.norm(self.k_vectors, axis=0) + 1e-10)
        self.logger.info("k-space grid computed successfully.")
        return khat

#----------------------------------------------------------------------------------------------------------------#

class ComputeFields(FourierSpaceManager):
    """
    A class to compute various fields such as the F0(k) and F2(k) fields from a 
    given FKP catalog, including constructing corresponding real-space fields 
    (Fx fields) and shell fields for Fourier analyses.

    This class extends the `FourierSpaceManager` to include specific operations 
    related to FKP catalog-based fields in cosmological surveys.

    Attributes:
    -----------
    fkp_catalog : FKPCatalog
        The FKP catalog containing data on positions, weights, and selection.
    config : dict
        Configuration dictionary containing the following keys:
        {
            'N': int,
            'L': float,
            'dk': float,
            'kmin': float,
            'kmax': float,
            'poles': List[int]
        }
        It is inherited from the parent class (FourierSpaceManager).
    num_triangles_path : str
        Path to the (.npy) file containing the number of triangles per bin. Defaults to None.
        If not None, the file is loaded and used to normalize the bispectrum.
        Otherwise, the triangle field is constructed to compute the number of triangles per bin.
    BoxCenter : array-like
        The center of the box in real space. Defaults to None.
    dtype : str
        Data type for mesh construction. Defaults to 'c16' (complex128).
    resampler : str
        Resampling method for constructing the mesh ('pcs', 'cic', etc.). Defaults to 'pcs'.
    compensated : bool
        Whether to apply compensation in the mesh. Defaults to True.
    interlaced : bool
        Whether to use interlacing for reducing aliasing effects in the mesh. Defaults to True.
    comp_weight : str
        The key for the compensation weight column in the FKP catalog. Defaults to 'WEIGHT'.
    fkp_weight : str
        The key for the FKP weight column in the FKP catalog. Defaults to 'FKPWeight'.
    selection : str
        The key for the selection mask column in the FKP catalog. Defaults to 'Selection'.
    position : str
        The key for the column containing the positions of objects in the FKP catalog. Defaults to 'POSITION'.
    F0k_field : np.ndarray
        The Fourier space F0(k) field.
    F2k_field : np.ndarray
        The Fourier space F2(k) field (None if only monopole is to be computed).
    k_vectors : np.ndarray
        The k-space vectors.
    k_magnitude : np.ndarray
        The magnitudes of the k-vectors.
    k_bins : list
        The bin centers for k-space magnitudes.
    k_bin_indices : list
        The indices of k-vectors that fall within each k-bin.
    F0x_field : np.ndarray
        The real-space field corresponding to F0(k).
    F2x_field : np.ndarray
        The real-space field corresponding to F2(k) (None if only monopole is to be computed).
    k_configs : list
        The valid k-space configurations based on the k-bins.
    shell_field : np.ndarray
        The shell field used to compute the number of triangles per bin.
    fkp_norm : float
        The FKP normalization factor (I22 & I33).

    """

    def __init__(self, fkp_catalog: FKPCatalog, config: dict, Fk_field: bool=True, Fx_field: bool=True,
                 num_triangles_path: Union[str, None] = None, BoxCenter=None, dtype='c16',
                 resampler='pcs', compensated=True, interlaced=True,
                 comp_weight='WEIGHT', fkp_weight='FKPWeight',
                 selection='Selection', position='POSITION') -> None:
        
        # Call the parent class constructor
        super().__init__(config)
        self.fkp_catalog = fkp_catalog
        self.num_triangles_path = num_triangles_path
        self.BoxCenter = BoxCenter
        self.dtype = dtype
        self.resampler = resampler
        self.compensated = compensated
        self.interlaced = interlaced
        self.comp_weight = comp_weight
        self.fkp_weight = fkp_weight
        self.selection = selection
        self.position = position

        self.logger = logging.getLogger(self.__class__.__name__)

        self.fkp_catalog['randoms/WEIGHT'] = np.ones(len(self.fkp_catalog['randoms/NZ'])) if 'randoms/WEIGHT' not in self.fkp_catalog.columns else self.fkp_catalog['randoms/WEIGHT']
        self.fkp_catalog['data/WEIGHT'] = np.ones(len(self.fkp_catalog['data/NZ'])) if 'data/WEIGHT' not in self.fkp_catalog.columns else self.fkp_catalog['data/WEIGHT']

        self.fkp_catalog['randoms/Selection'] = np.ones(len(self.fkp_catalog['randoms/NZ']), dtype=bool) if 'randoms/Selection' not in self.fkp_catalog.columns else self.fkp_catalog['randoms/Selection']
        self.fkp_catalog['data/Selection'] = np.ones(len(self.fkp_catalog['data/NZ']), dtype=bool) if 'data/Selection' not in self.fkp_catalog.columns else self.fkp_catalog['data/Selection']

        # Initialize the F0(k) field and k-space properties
        self.F0k_field = self.construct_F_field() if Fk_field else None
        self.k_vectors, self.k_magnitude = self.k_grid()

        # Construct F2(k) if needed (for poles other than monopole)
        if self.poles == [0]:
            self.F2k_field = None
        else:
            self.F2k_field = self.construct_F2k_field(fkp_field=self.fkp_real_field, F0k_field=self.F0k_field) if Fk_field else None

        # Correct for window function in the fields
        p_value = {
            'ngp': 1, 
            'cic': 2, 
            'tsc': 3, 
            'pcs': 4
        }

        if Fk_field:
            self.logger.info("Deconvolving the window function from the F(k)-fields...")
            self.F0k_field /= self.Window_function(p=p_value[self.resampler])
            if self.F2k_field is not None:
                self.F2k_field /= self.Window_function(p=p_value[self.resampler])
            self.logger.info("Window function deconvolution successfull")

        # Get k-bins and k-bin indices
        self.k_bins, self.k_bin_indices = self.get_k_bin_indices(self.k_magnitude)
        
        if Fx_field:
            # Construct Fx-field for l=0
            self.logger.info('Computation for l=0...')
            self.F0x_field = self.construct_shell_field(substitute=self.F0k_field, field_name='F0x field')

            # Construct F2(x) field if F2(k) is available
            if self.F2k_field is not None:
                self.logger.info("Computation for l=2...")
                self.F2x_field = self.construct_shell_field(substitute=self.F2k_field, field_name='F2x field')
            else:
                self.F2x_field = None

        # Construct k-configurations
        self.k_configs = self.create_k_configurations(self.k_bins)
                                                                                                      
        # Compute I33 normalization 
        try: 
            self.I33_data, self.I33_randoms = self.fkp_norm(self.fkp_catalog, norm_type='I33')
        except Exception as e:
            self.logger.info("I33 normalization not computed. Defaulting to 1.0")
            self.logger.error(f"An error occurred: {e}")
            self.I33_data, self.I33_randoms = 1.0, 1.0
        
        # Compute I22 normalization 
        try: 
            self.I22_data, self.I22_randoms = self.fkp_norm(self.fkp_catalog, norm_type='I22')
        except Exception as e:
            self.logger.info("I22 normalization not computed. Defaulting to 1.0")
            self.logger.error(f"An error occurred: {e}")
            self.I22_data, self.I22_randoms = 1.0, 1.0
        
        # Compute Shotnoise for 2-point statistics
        try: 
            self.logger.info("Computing shot noise for 2-point statistics...")
            self.N0_data, self.N0_randoms = self.shotnoise_2pt(self.fkp_catalog, self.I22_randoms)
            self.logger.info("Shot noise for 2-point statistics computed successfully.")
        except Exception as e:
            self.logger.warning("Shot noise for 2-point statistics not computed. Defaulting to 1.0")
            self.logger.error(f"An error occurred: {e}")
            self.N0_data, self.N0_randoms = 1.0, 1.0

        # Compute Shotnoise for 3-point statistics
        try:
            self.logger.info("Computing shot noise for 3-point statistics...")
            self.N0_3pt, self.N2_3pt = self.shotnoise_3pt(self.fkp_catalog, self.I33_randoms).values()
            self.logger.info("Shot noise for 3-point statistics computed successfully.")
        except Exception as e:
            self.logger.warning("Shot noise for 3-point statistics not computed. Defaulting to 1.0")
            self.logger.error(f"An error occurred: {e}")
            self.N0_3pt, self.N2_3pt = 1.0, 1.0

        assert abs(self.I22_randoms - self.I22_data) / self.I22_randoms * 100 < 3, "I22 normalization mismatch between data and randoms is greater than 3%. Check the catalogs/selections. I22.data: {self.I22_data}, I22.randoms: {self.I22_randoms}"

        # Create a dictionary to store the attributes
        self.properties = {
            'Nmesh': self.N,
            'BoxSize': self.L,
            'dk': self.dk,
            'kmin': self.kmin,
            'kmax': self.kmax,
            'poles': self.poles,
            'alpha': self.fkp_catalog.attrs['alpha'],
            'fsky': self.fkp_catalog.attrs['fsky'],
            'P0': self.fkp_catalog.attrs['P0'],
            'I22.data': self.I22_data,
            'I22.randoms': self.I22_randoms,
            'I33.data': self.I33_data,
            'I33.randoms': self.I33_randoms,
            'shotnoise2pt.data': self.N0_data,
            'shotnoise2pt.randoms': self.N0_randoms,
            'shotnoise2pt': self.N0_randoms + self.N0_data,
            'shotnoise3pt_N0': self.N0,
            'shotnoise3pt_0': self.N0_3pt,
            'shotnoise3pt_2': self.N2_3pt,
            'resampler': self.resampler,
            'dtype': self.dtype,
            'interlaced': self.interlaced,
            'compensated': self.compensated,
            'k_bins': self.k_bins
        }

        if self.BoxCenter is not None:
            self.properties.update({'BoxCenter': list(self.BoxCenter)})

    @lru_cache(maxsize=None, typed=False)
    def Window_function(self, p: int) -> np.ndarray:
        """
        Computes the window function in k-space for correcting the resolution of the grid.
        
        This function applies a sinc filter to each dimension in k-space (kx, ky, kz) with an exponent
        p that controls the degree of filtering.

        Parameters:
        -----------
        p : int
            The exponent used to modify the window function. Higher values (e.g., p=4) reduce aliasing
            and sharpen the k-space resolution.

        Returns:
        --------
        np.ndarray
            The window function in k-space, applied to the kx, ky, and kz components.
        """
        self.logger.info("Computing Window function...")
        kx, ky, kz = self.k_vectors
        ks = 2 * np.pi / (self.L / self.N)
        wx = (np.sinc(kx / ks)) ** p
        wy = (np.sinc(ky / ks)) ** p
        wz = (np.sinc(kz / ks)) ** p
        self.logger.info("Window function computed successfully.")
        return wx * wy * wz

    @profile
    def construct_F_field(self) -> np.ndarray:
        """
        Constructs the F0(k) field in Fourier space by transforming the FKP catalog data to a real field 
        and performing a Fourier transform.

        This method first generates a real-space mesh from the FKP catalog using the configured resampling 
        method, compensating for any biases. The field is then Fourier transformed to get the F0(k) field.
        
        Returns:
        --------
        np.ndarray
            The Fourier transformed F0(k) field, which is a 3D array in k-space.
        """
        if self.L is None:
            self.L = np.round(np.max(self.fkp_catalog['data/POSITION'].compute()))
            self.config['L'] = self.L

        self.logger.info("Constructing F0(k) field...")
        
        # Create FKP mesh (Comment out the following line to check for the reference bispectrum data)
        fkp_mesh = self.fkp_catalog.to_mesh(Nmesh=self.N, BoxSize=self.L, BoxCenter=self.BoxCenter, dtype=self.dtype,
                                            resampler=self.resampler, compensated=self.compensated, interlaced=self.interlaced,
                                            comp_weight=self.comp_weight, fkp_weight=self.fkp_weight,
                                            selection=self.selection, position=self.position)
        self.BoxCenter = fkp_mesh.attrs['BoxCenter']
        
        if self.N is None:
            self.N = fkp_mesh.attrs['Nmesh']
            self.config['N'] = self.N

        if self.k_nyquist is None:
            self.dk = 2 * np.pi / self.L if self.dk is None else self.dk
            self.config['dk'] = self.dk

            if self.kmin is None:
                self.kmin = 0 + self.dk/2
                self.config['kmin'] = self.kmin

            self.k_nyquist = np.pi * self.N / self.L
            self.kmax = self.k_nyquist
            self.config['kmax'] = self.kmax
            
            self.k_vectors, self.k_magnitude = self.k_grid()
        
        self.logger.info(f"knyquist: {self.k_nyquist}, dk: {self.dk}, kmin: {self.kmin}, kmax: {self.kmax}")

        self.logger.info(f"Nmesh: {self.N}")
        self.logger.info(f"BoxSize: {self.L}")

        # Convert the mesh to a real field and normalize
        self.logger.info("Converting mesh into the real-field...")
        self.fkp_real_field = fkp_mesh.to_real_field()

        self.fkp_real_field *= self.L ** 3 / self.N ** 3
        self.logger.info("Real field constructed successfully. Scaled the field by the grid volume.")

        self.logger.info("Computing the Fourier transform of the real field...")
        F0k_field = np.fft.fftn(self.fkp_real_field)
        self.logger.info("F0(k) field constructed successfully.")

        return F0k_field
    
    @profile
    def construct_F2k_field(self, fkp_field, F0k_field) -> np.ndarray:
        """
        Constructs the F2(k) field, which corresponds to the quadrupole term in Fourier space.
        
        This method calculates the quadrupole (l=2) term by combining the real-space field components
        and transforming them to k-space using the Fast Fourier Transform (FFT).
        
        Returns:
        --------
        np.ndarray
            The Fourier transformed F2(k) field, which is a 3D array in k-space.
        """
        self.logger.info("Constructing F2(k) field...")
        xhat = self.x_hat()
        khat = self.k_hat()
        
        components = list(itertools.combinations_with_replacement(range(3), 2))
        F_first_term = np.complex128(0)

        for i, j in components:
            Q_ij = fkp_field * xhat[i] * xhat[j]
            Q_ij = np.fft.fftn(Q_ij)

            if i != j:
                F_first_term += 2 * khat[i] * khat[j] * Q_ij
            else:
                F_first_term += khat[i] * khat[j] * Q_ij

        del Q_ij, xhat, khat
        
        F2_k_field = (1.5*F_first_term - 0.5*F0k_field)
        self.logger.info("F2(k) field constructed successfully.")
        return F2_k_field

    @profile  
    def construct_shell_field(self, substitute: Union[np.ndarray, int, float]=1, field_name: str="Shell field") -> np.ndarray:
        """
        Constructs a shell field in k-space, and takes the inverse Fourier transform to 
        convert it back to real space. This method assigns a specified value to each shell 
        of k-space within the selected k-bins.

        Parameters:
        -----------
        substitute : Union[np.ndarray, int, float]
            The value (or array of values) to assign to each k-bin. If an array, it must match the shape of the grid.
        
        field_name : str
            The name of the field for logging purposes.

        Returns:
        --------
        np.ndarray
            A 4D array with the shape `(len(k_bins), N, N, N)`, where each 3D slice represents 
            a k-bin in real space after the inverse FFT.
        """
        self.logger.info(f"Constructing {field_name}...")
        shell_field = np.zeros((len(self.k_bins), self.N, self.N, self.N), dtype=np.complex128)
        for k_bin_index in range(len(self.k_bins)):
            indices = self.k_bin_indices[k_bin_index]
            if np.any(indices):
                if isinstance(substitute, np.ndarray) and substitute.shape != (self.N, self.N, self.N):
                    raise ValueError(f"Expected substitute array of shape {(self.N, self.N, self.N)}, but got {substitute.shape}")
                elif isinstance(substitute, (int, float)):
                    shell_field[k_bin_index][indices] = substitute
                else:
                    shell_field[k_bin_index][indices] = substitute[indices]
                shell_field[k_bin_index] = np.fft.ifftn(shell_field[k_bin_index]).real
            else:
                self.logger.error(f"No indices found in k-bin {self.k_bins[k_bin_index]}. This can is due to incorrect configuration of number of grid points and box size.")
                raise ValueError(f"No indices found in k-bin {self.k_bins[k_bin_index]}")
        
        self.logger.info(f"{field_name} constructed successfully.")
        return shell_field
    
    @profile
    def num_triangles(self) -> np.ndarray:
        """
        Computes the number of triangles for each k-bin. If a precomputed file exists at `num_triangles_path`,
        it will load the data; otherwise, it constructs the shell field and calculates the product to determine 
        the number of triangles.

        Returns:
        --------
        np.ndarray
            An array containing the number of triangles for each k-bin.

        Raises:
        -------
        FileNotFoundError
            If `num_triangles_path` is set but the file is not found.
        """

        if self.num_triangles_path is None:
            triangle_field = self.construct_shell_field(substitute=1, field_name="Triangle field")
            self.logger.info("Computing the number of triangles per bin...")
            self.num_triangles_per_bin = fields_product(triangle_field, triangle_field, triangle_field, self.k_configs)
            del triangle_field
            self.logger.info("Number of triangles per bin computed successfully.")
        else:
            try:
                self.num_triangles_per_bin = np.load(self.num_triangles_path)
                self.logger.info("Number of triangles per bin loaded successfully.")
            except Exception as e:
                self.logger.error(f"An error occurred: {e}")
                raise e
            except FileNotFoundError as e:
                self.logger.error(f"File not found: {e}")
                raise e
        return self.num_triangles_per_bin
    
    @profile
    def shotnoise_2pt(self, fkp_catalog: FKPCatalog, I22: float=1.0) -> Tuple[float, float]:
        """
        Computes the shot noise term for the 2-point correlation function in Fourier space.
        This accounts for the Poisson noise contribution to the two-point statistics.

        Returns:
        --------
        np.ndarray 
            The computed shot noise term, normalized by the I33 factor for each k-bin.
        """
        N0_data = np.sum(fkp_catalog['data/WEIGHT']**2 * fkp_catalog['data/FKPWeight']**2) / I22
        N0_randoms = fkp_catalog.attrs['alpha']**2 * np.sum(fkp_catalog['randoms/WEIGHT']**2 * fkp_catalog['randoms/FKPWeight']**2) / I22
        return N0_data.compute(),  N0_randoms.compute()
    
    @profile
    def shotnoise_3pt(self, fkp_catalog: FKPCatalog, I33: float=1.0) -> Dict[str, np.ndarray]:
        """
        Computes the shot noise term for the 3-point correlation function in Fourier space.
        This accounts for the Poisson noise contribution to the three-point statistics.

        Returns:
        --------
        dict[str, ndarray]
            A dictionary containing the shot noise terms for the 3-point statistics for each pole.
        """
        self.N0 = ((np.sum(fkp_catalog['data/FKPWeight']**3) - fkp_catalog.attrs['alpha']**3 * np.sum(fkp_catalog['randoms/FKPWeight']**3)) / I33).compute()

        fkp_catalog['data/FKPWeight_SN'] = fkp_catalog['data/FKPWeight']**2
        fkp_catalog['randoms/FKPWeight_SN'] = -1 * fkp_catalog.attrs['alpha'] * fkp_catalog['randoms/FKPWeight']**2
        
        fkp_mesh_SN = fkp_catalog.to_mesh(Nmesh=self.N, BoxSize=self.L, BoxCenter=self.BoxCenter, dtype=self.dtype,
                                            resampler=self.resampler, compensated=self.compensated, interlaced=self.interlaced,
                                            comp_weight=self.comp_weight, fkp_weight='FKPWeight_SN',
                                            selection=self.selection, position=self.position)

        fkp_real_field_SN = fkp_mesh_SN.to_real_field()
        fkp_real_field_SN *= self.L ** 3 / self.N ** 3

        p_value = {
            'ngp': 1, 
            'cic': 2, 
            'tsc': 3, 
            'pcs': 4
        }
        
        self.logger.info("Computing the weighted F0k field for shot noise...")
        self.F0w = np.fft.fftn(fkp_real_field_SN)
        self.F0w /= self.Window_function(p=p_value[self.resampler])
        self.logger.info("Weighted F0k field computed successfully.")

        del fkp_mesh_SN

        self.logger.info("Constructing weighted F2k field for shot noise...")
        self.F2w = self.construct_F2k_field(fkp_field=fkp_real_field_SN, F0k_field=self.F0w)
        self.F2w /= self.Window_function(p=p_value[self.resampler])
        self.logger.info("Weighted F2k field constructed successfully.")

        def spherical_average(field: np.ndarray, k_bin_info: dict) -> np.ndarray:
            """
            Computes the spherical average of a field within each k-bin.

            This method averages the input field over specified k-bins defined by the `k_bin_indices` attribute.

            Parameters:
            -----------
            field : np.ndarray
                Input field to average over the defined k-bins.

            Returns:
            --------
            np.ndarray
                Array of spherical averages, one for each k-bin.
            """
            spherical_averages = []
            for k_bin, indices in k_bin_info.items():
                if (len(indices[0]) > 0): 
                    # Compute the average of the values for specific k-bin
                    spherical_averages.append(np.mean(field[indices]))
                else:
                    # If there are no indices in this bin, set the average to NaN
                    spherical_averages.append(np.nan)
            return np.array(spherical_averages).real
        
        shotnoise_3pt_dict={}
        
        # Power spectrum of the fields for monopole shotnoise
        SN_term_k123_l0 = spherical_average(self.F0k_field*self.F0w, self.k_bin_info) / I33
        
        # Power spectrum of fields for quadrupole shotnoise
        SN_term_k1_l2 = 5 * spherical_average(self.F2k_field*self.F0w, self.k_bin_info) / I33
        SN_term_k2_k3_l2 = 5 * spherical_average(self.F0k_field*self.F2w, self.k_bin_info) / I33
        
        def cosine(k1, k2, k3):
            return (k1**2 + k2**2 - k3**2)/(2*k1*k2)
        
        # Compute the shot noise for 3-point statistics
        SN_3pt_0, SN_3pt_2 = [], []

        # Precompute the Legendre polynomials for l=2
        cosines_k2 = np.array([cosine(k1, k2, k3) for (k1, k2, k3) in self.k_configs])
        cosines_k3 = np.array([cosine(k1, k3, k2) for (k1, k2, k3) in self.k_configs])

        L_k2_2 = legendre(2)(cosines_k2)
        L_k3_2 = legendre(2)(cosines_k3)

        for idx, (k1, k2, k3) in enumerate(self.k_configs):

            SN_3pt_0.append(SN_term_k123_l0[k1-1]  +  SN_term_k123_l0[k2-1]  +  SN_term_k123_l0[k3-1])
            SN_3pt_2.append(SN_term_k1_l2[k1-1]   +   L_k2_2[idx] * SN_term_k2_k3_l2[k2-1]   +   L_k3_2[idx] * SN_term_k2_k3_l2[k3-1])

        SN_3pt_0 = np.array(SN_3pt_0)
        SN_3pt_2 = np.array(SN_3pt_2)

        shotnoise_3pt_dict['0'] = SN_3pt_0 - (2*self.N0)
        shotnoise_3pt_dict['2'] = SN_3pt_2

        del fkp_real_field_SN

        return shotnoise_3pt_dict

    @profile
    def fkp_norm(self, fkp_catalog: FKPCatalog, norm_type: str='I33') -> Tuple[float, float]:
        """
        Computes the normalization for the FKP weights. This normalizes the fields to ensure 
        that the Fourier transformed fields are properly weighted.
        
        Returns:
        --------
        np.ndarray
            The normalization factor for each k-bin, computed using I33 for Fourier space normalization.
        """

        if norm_type == 'I33':
            self.logger.info("Computing FKP normalization (I33)...")
            
            I_33_data = (np.sum(fkp_catalog['data/NZ']**2 * fkp_catalog['data/WEIGHT']**2 \
                                                            * fkp_catalog['data/FKPWeight']**3)).compute()
            
            I_33_randoms = (fkp_catalog.attrs['alpha'] * np.sum(fkp_catalog['randoms/NZ']**2 * fkp_catalog['randoms/WEIGHT']**2 \
                                                            * fkp_catalog['randoms/FKPWeight']**3)).compute()

            self.logger.info("I33 computed successfully.")
            return I_33_data, I_33_randoms
        
        elif norm_type == 'I22':
            self.logger.info("Computing FKP normalization (I22)...")
            
            I_22_data = (np.sum(fkp_catalog['data/NZ'] * fkp_catalog['data/WEIGHT'] * fkp_catalog['data/FKPWeight']**2)).compute()

            I_22_randoms = (fkp_catalog.attrs['alpha'] * np.sum(fkp_catalog['randoms/NZ'] * fkp_catalog['randoms/WEIGHT'] \
                                                            * fkp_catalog['randoms/FKPWeight']**2)).compute()
            self.logger.info("I22 computed successfully.")
            return I_22_data, I_22_randoms
        else:
            self.logger.error("Invalid normalization type. Choose from 'I33' or 'I22'")
            raise ValueError("Invalid normalization type. Choose from 'I33' or 'I22'")

#----------------------------------------------------------------------------------------------------------------#

@nb.njit(parallel=True)
def fields_product(field_arr_l: np.ndarray, field_arr1: np.ndarray, 
                   field_arr2: np.ndarray, conf: np.ndarray) -> np.ndarray:
    """
    Computes the product of fields for bispectrum calculation in a loop over 
    k-space configurations, parallelized for efficiency.

    Parameters:
    -----------
    field_arr_l : np.ndarray
        The first field array, which could either be F0(x) field or F2(x) field.
    field_arr1 : np.ndarray
        The second field array, which is the F0(x) fields).
    field_arr2 : np.ndarray
        The third field array, which is the F0(x) fields).
    conf : np.ndarray
        The k-space configuration array (indices for triads of k-values).

    Returns:
    --------
    np.ndarray:
        The product of fields over all configurations, summed across relevant indices.
    """
    result = np.zeros(conf.shape[0], dtype=field_arr1.dtype)
    for i in nb.prange(conf.shape[0]):
        result[i] = np.sum(field_arr_l[conf[i, 0] - 1] * field_arr1[conf[i, 1] - 1] * field_arr2[conf[i, 2] - 1])
    return result.real

#----------------------------------------------------------------------------------------------------------------#

class Bispectrum(ComputeFields):
    """
    A class for bispectrum calculation using FKP-based fields in cosmology, 
    extending the `ComputeFields` class to compute and save bispectrum results.
    """

    def __init__(self, Fields: ComputeFields) -> None:
        """
        Initialize the Bispectrum class by setting up parameters inherited from `ComputeFields` 
        and initializing fields specific to bispectrum calculation.

        Parameters:
        -----------
        Fields : ComputeFields
            An instance of the ComputeFields class, which provides k-space configurations, fields, 
            and other necessary data for bispectrum computation.

        Attributes:
        -----------
        L : float
            The size of the box in real space.
        N : int
            The number of grid points in each dimension.
        k_vectors : np.ndarray
            The k-space vectors.
        k_bins : np.ndarray
            The k-space bin edges.
        k_bin_indices : np.ndarray
            The k-space bin indices.
        k_configs : np.ndarray
            The k-space configurations for triads.
        poles : list
            The list of poles to be used for bispectrum computation.
        F0x_field : np.ndarray
            The real-space monopole F-field.
        F2x_field : np.ndarray
            The real-space quadrupole F-field (if applicable).
        I33 : np.ndarray
            The normalization factor for the bispectrum computation.
        num_triangles_per_bin : np.ndarray
            The number of triangles for each k-bin.
        """
        
        self.L = Fields.L
        self.N = Fields.N
        self.k_vectors = Fields.k_vectors
        self.k_bins = Fields.k_bins
        self.k_bin_indices = Fields.k_bin_indices
        self.k_configs = Fields.k_configs
        self.poles = Fields.poles
        self.resampler = Fields.resampler

        self.F0x_field = Fields.F0x_field
        self.F2x_field = Fields.F2x_field
        self.I33 = Fields.I33_randoms
        
        self.logger = logging.getLogger(self.__class__.__name__)
        self.num_triangles_per_bin = Fields.num_triangles()

    @profile
    def compute(self) -> np.ndarray: 
        """
        Computes the bispectrum for the specified poles (monopole and quadrupole) using 
        the FKP-based fields.

        The bispectrum is computed for each pole, normalized by the number of triangles 
        and the I33 factor.

        Returns:
        --------
        dict
            A dictionary where the keys represent the poles (0 for monopole, 2 for quadrupole),
            and the values are the corresponding bispectrum results.
        """

        # Loop over poles and compute bispectrum
        self.bispectrum_results = {}
        for pole in tqdm(self.poles):
            if pole == 0:
                # Compute monopole bispectrum
                self.logger.info(f"Computing monopole bispectrum for pole: {pole}")
                bispec_mono = fields_product(self.F0x_field, self.F0x_field, self.F0x_field, self.k_configs)
                self.logger.info(f"Bispectrum computation successfull for pole: {pole}")
                bispec_mono /= self.num_triangles_per_bin # Normalize by the number of triangles
                bispec_mono /= self.I33 # Normalize by I33
                self.bispectrum_results[f'pole_{pole}'] = bispec_mono
            else:
                # Compute quadrupole bispectrum
                self.logger.info(f"Computing quadrupole bispectrum for pole: {pole}")
                bispec_quad = fields_product(self.F2x_field, self.F0x_field, self.F0x_field, self.k_configs)
                self.logger.info(f"Bispectrum computation successfull for pole: {pole}")
                bispec_quad /= self.num_triangles_per_bin # Normalize by the number of triangles
                bispec_quad /= self.I33 # Normalize by I33
                self.bispectrum_results[f'pole_{pole}'] = 5 * bispec_quad
        self.logger.info("Bispectrum computation successfull. Normalized by I33 and number of triangles.")
        return self.bispectrum_results
    
    def save(self, file_path: str) -> None:
        """
        Saves the bispectrum results to a specified directory. 

        The bispectrum results are saved as `.npy` files, where the file name includes 
        the pole (e.g., 0 for monopole, 2 for quadrupole) and the species name.

        Parameters:
        -----------
        save_path : str
            The directory path where the bispectrum results will be saved. If the directory does not exist, 
            it will be created.
        
        species : str
            The species name to be included in the file name.
        
        Returns:
        --------
        None
        """
        self.logger.info("Saving bispectrum values...")

        if file_path is not None:
            try:
                with open(file_path, 'wb') as f:
                    pickle.dump(self.bispectrum_results, f)
                self.logger.info("All bispectrum terms saved successfully.")
            except IOError as e:
                self.logger.error(f"Failed to save file at {file_path}: {e}")
                raise e
        else:
            self.logger.error("No file path provided. Please provide a valid file path.")
            raise ValueError("No file path provided. Please provide a valid file path.")
        


class PowerSpectrum(ComputeFields):
    """
    A class for power spectrum calculation using FKP-based fields in cosmology, 
    extending the `ComputeFields` class to compute and save power spectrum results.
    """

    def __init__(self, Fields: ComputeFields) -> None:
        """
        Initialize the PowerSpectrum class by setting up parameters inherited from `ComputeFields` 
        and initializing fields specific to power spectrum calculation.

        Parameters:
        -----------
        Fields : ComputeFields
            An instance of the ComputeFields class, which provides k-space configurations, fields, 
            and other necessary data for power spectrum computation.

        Attributes:
        -----------
        k_bins : np.ndarray
            The k-space bin edges.
        k_bin_indices : np.ndarray
            The k-space bin indices.
        poles : list
            The list of poles to be used for power spectrum computation.
        F0k_field : np.ndarray
            The fourier-space monopole F-field.
        F2k_field : np.ndarray
            The fourier-space quadrupole F-field (if applicable).
        I22 : np.ndarray
            The normalization factor for the power spectrum computation.
        """

        self.k_bins = Fields.k_bins
        self.properties = Fields.properties
        self.k_bin_info = Fields.k_bin_info
        self.poles = Fields.poles
        self.F0k_field = Fields.F0k_field
        self.F2k_field = Fields.F2k_field
        self.I22 = Fields.I22_randoms
        
        self.logger = logging.getLogger(self.__class__.__name__)

    def spherical_average(self, field: np.ndarray) -> np.ndarray:
        """
        Computes the spherical average of a field within each k-bin.

        This method averages the input field over specified k-bins defined by the `k_bin_indices` attribute.

        Parameters:
        -----------
        field : np.ndarray
            Input field to average over the defined k-bins.

        Returns:
        --------
        np.ndarray
            Array of spherical averages, one for each k-bin.
        """
        spherical_averages = []
        for k_bin, indices in self.k_bin_info.items():
            if (len(indices[0]) > 0): 
                # Compute the average of the values for specific k-bin
                spherical_averages.append(np.mean(field[indices]))
            else:
                # If there are no indices in this bin, set the average to NaN
                spherical_averages.append(np.nan)
        return np.array(spherical_averages).real
    
    @profile
    def compute(self) -> np.ndarray: 
        """
        Computes the power spectrum for the specified poles (monopole and quadrupole) using 
        the FKP-based fields.

        The power spectrum is computed for each pole, normalized by I22.

        Returns:
        --------
        dict
            A dictionary where the keys represent the poles (0 for monopole, 2 for quadrupole),
            and the values are the corresponding power spectrum results.
        """

        # Loop over poles and compute power spectrum
        self.power_spectrum_results = {}
        for pole in tqdm(self.poles):
            if pole == 0:
                # Compute monopole power spectrum
                self.logger.info(f"Computing monopole power spectrum for pole: {pole}")
                power_mono = self.spherical_average(np.abs(self.F0k_field)**2) / self.I22 - self.properties['shotnoise2pt']
                self.logger.info(f"Power spectrum computation successfull for pole: {pole}")
                self.power_spectrum_results[f'pole_{pole}'] = power_mono
            else:
                # Compute quadrupole power spectrum
                self.logger.info(f"Computing quadrupole power spectrum for pole: {pole}")
                power_quad = 5 * self.spherical_average(np.conjugate(self.F0k_field) * self.F2k_field) / self.I22
                self.logger.info(f"Power spectrum computation successfull for pole: {pole}")
                self.power_spectrum_results[f'pole_{pole}'] = power_quad
        
        self.power_spectrum_results['k_bins'] = self.k_bins
        
        self.logger.info("Power spectrum computation successfull. Normalized by I22.")
        return self.power_spectrum_results
    
    def save(self, file_path: str) -> None:
        """
        Saves the power spectrum results to a specified directory. 

        The power spectrum results are saved as `.npy` files, where the file name includes 
        the pole (e.g., 0 for monopole, 2 for quadrupole) and the species name.

        Parameters:
        -----------
        save_path : str
            The directory path where the power spectrum results will be saved. If the directory does not exist, 
            it will be created.
        
        species : str
            The species name to be included in the file name.
        
        Returns:
        --------
        None
        """
        self.logger.info("Saving power spectrum values...")

        if file_path is not None:
            try:
                with open(file_path, 'wb') as f:
                    pickle.dump(self.power_spectrum_results, f)
                self.logger.info("All power spectrum terms saved successfully.")
            except IOError as e:
                self.logger.error(f"Failed to save file at {file_path}: {e}")
                raise e
        else:
            self.logger.error("No file path provided. Please provide a valid file path.")
            raise ValueError("No file path provided. Please provide a valid file path.")
        
#----------------------------------------------------------------------------------------------------------------#

class BispectrumTerms:
    """

    A class for calculating bispectrum terms, including auto- and cross-terms, for a set of species 
    based on specified triangle configurations and poles. The bispectrum terms are used to analyze 
    interactions between different species in a dataset.


    Attributes
    ----------
    species_names : List[str]
        List of species names involved in the computation. Each species name must have a unique first letter, 
        as the first letter is used to create short names for file naming and mapping purposes.
    num_triangles_per_bin : np.ndarray
        Array specifying the number of wavenumber triangle configurations in each bin, 
        used the normalization of bispectrum terms.
    k_configs : np.ndarray
        Array defining wavenumber triangle configurations in Fourier space, 
        which are used to compute bispectrum terms based on different triangle shapes and sizes.
    cat_properties_dir : str
        Path to the directory containing catalog properties files.
    logger : logging.Logger
        Logger for the class, initialized based on the class name.
    map_name : dict
        Mapping from species names to a short, lowercase version of the name, 
        used for file naming and referencing in calculations.
    species_combs : list
        List of all unique combinations of species pairs, used to compute cross-terms. 
    F_poles : list
        List of poles (e.g., ['F0_x', 'F2_x']) to be computed, where each pole corresponds 
        to a specific term in the Fourier transform (e.g., monopole for F0_x, quadrupole for F2_x).
    
    Methods
    -------
    import_fields(Fx_save_dir: str, workers: int = os.cpu_count())
        Loads Fourier-space fields for each species from specified directory and returns them as a dictionary.
    compute_single_combination(t1, t2, t3, F0x, F2x, cross_term_name)
        Computes monopole and optionally quadrupole terms for a single combination of species fields.
    compute_autoterms(F0x_species, F2x_species)
        Computes auto-terms for a given species based on Fourier-space fields.
    process_pair(Fx_files: dict, species: List[str])
        Processes a pair of species and computes cross-terms, returning them as a dictionary.
    compute(Fx_files: dict, file_save_path: str = None, workers: int = os.cpu_count())
        Computes bispectrum terms for all species combinations and saves results to a file if specified.
    
    """
    def __init__(self, species_names: List[str], num_triangles_per_bin: np.ndarray, k_configs: np.ndarray, I33: dict, poles: List[int]=[0]) -> None:
        """
        Initializes the class with species names, number of triangles per bin, k-space configurations, 
        and the list of poles to compute. Validates input by ensuring species names are unique in their 
        first letters and that there are multiple species for cross-term computation. Creates combinations 
        for species pairs if applicable.
        """
        self.species_names = species_names
        self.num_triangles_per_bin = num_triangles_per_bin
        self.k_configs = k_configs
        self.I33 = I33

        assert all(species in list(I33.keys()) for species in species_names) == True, "I33 values not provided for all species."
        assert 'total' in I33.keys(), "I33 value not provided for the combined catalog. Expected key: 'total'."

        assert len(self.num_triangles_per_bin) == len(self.k_configs)
        assert len(self.species_names) > 0

        self.logger = logging.getLogger(self.__class__.__name__)

        # Convert species names to lowercase
        self.species_names = list(map(str.lower, self.species_names))

        # Check if the first letter of the species name is unique
        if len(set([name[0] for name in self.species_names])) != len(self.species_names):
            self.logger.error("The first letter of the species name is not unique. Choose different names.")
            raise ValueError("The first letter of the species name is not unique.")
    
        # Mapping file name suffix to a short species name
        self.map_name = {name: name[0].lower() for name in self.species_names}
        self.logger.info("Name mapping: %s", self.map_name)

        # Generate combinations of species
        if len(self.species_names) > 1:
            self.species_combs = list(itertools.combinations(self.species_names, 2))
            self.species_combs = [sorted(species) for species in self.species_combs]
        else:
            self.logger.warning("Only auto-terms will be calculated since there is only one species.")
        self.logger.info("Species combinations: %s", self.species_combs)

        # Define poles to compute
        self.F_poles = ['F0_x']
        if 2 in poles:
            self.F_poles.append('F2_x')
        self.logger.info("Poles to compute: %s", self.F_poles)

        self.f_c = self.compute_species_fraction(self.I33, self.species_names)
        self.prefacts = {}

    def import_fields(self, Fx_save_dir: str, workers: int=None):
        """
        Loads Fourier-space fields for each species from files in the specified directory using parallel 
        processing. If the directory does not exist, raises a FileNotFoundError.

        Parameters
        ----------
        Fx_save_dir : str
            Path to the directory where Fourier-space field files are saved.
        workers : int, optional
            Number of workers for parallel file loading; defaults to CPU count.
        
        Returns
        -------
        dict
            Dictionary containing Fourier-space fields for each species and pole.
        """
        if os.path.exists(Fx_save_dir) is False:
            raise FileNotFoundError(f"Directory {Fx_save_dir} does not exist.")
        
        if workers is None:
            workers = os.cpu_count()

        # Load species files
        def load_file(file_path):
            """Helper function to load a single file."""
            return np.load(file_path, allow_pickle=False)
        
        filepaths = [os.path.join(Fx_save_dir, f"{F_pole}_{name}.npy") for F_pole in self.F_poles for name in self.species_names]

        pool_workers = min(workers, len(filepaths)) # Number of workers for parallel loading

        if pool_workers != workers:
            self.logger.info(f"Chose {pool_workers} workers for parallel loading to avoid thread contention.") 
        else:
            self.logger.info(f"Using {pool_workers} workers/threads for parallel loading.")

        # Parallel loading using ThreadPoolExecutor (multithreading)
        with ThreadPoolExecutor(max_workers=pool_workers) as executor:
            results = list(executor.map(load_file, filepaths))

        self.logger.info(f"{len(results)} files read successfully.")
        species_files = {f"{F_pole}_{name}": results[i] for i, (F_pole, name) in enumerate(itertools.product(self.F_poles, self.species_names))}
        self.logger.info("All files successfully loaded in a dictionary.")

        return species_files
    
    @profile
    def compute_species_fraction(self, I33: dict, species_names: List[str]) -> dict:
        """
        Computes the fraction for each species relative to the full catalog.

        This method calculates the fraction for each species using the provided I33 normalization values. 
        It normalizes by the I33 value of the combined catalog ('total').

        Parameters:
        -----------
        I33 : dict
            Dictionary of I33 normalization values for each species, including a key 'total' for the full catalog.
            
        species_names : List[str]
            List of species names to compute the fractions for.

        Returns:
        --------
        dict
            A dictionary mapping species names to their corresponding fractional value based on I33.
        """
        f_c = {}
        for species in species_names:
            f_c[f'fc.{species}'] = np.power(I33[species] / I33['total'], 1/3)
        return f_c
    
    @profile
    def compute_single_combination(self, t1: int, t2: int, t3: int, \
                                   F0x: List[np.ndarray], F2x: List[np.ndarray], norm: float, cross_term_name: str) -> dict:
        """
        Computes monopole and optionally quadrupole terms for a single combination of species fields. 
        The terms are computed based on the provided wavenumber configurations and poles.

        Parameters
        ----------
        t1, t2, t3 : int
            Indices representing the combination of species fields to compute.
        F0x, F2x : list of np.ndarray
            Fourier-space fields for monopole and quadrupole terms, respectively.
        cross_term_name : str
            Name for the computed cross term.
        
        Returns
        -------
        dict
            Dictionary containing monopole and quadrupole results for the specified combination.
        """
        result = {'monopole': None, 'quadrupole': None}

        self.logger.info('Computing cross term: %s', cross_term_name)
        
        # Compute monopole for this combination
        result['monopole'] = {cross_term_name: fields_product(F0x[t1], F0x[t2], F0x[t3], self.k_configs) / self.num_triangles_per_bin / norm}
        self.logger.info('Computed monopole for cross term: %s', cross_term_name)
        
        # If F2x fields are provided, compute quadrupole as well
        if F2x[0] is not None and F2x[1] is not None:
            result['quadrupole'] = {cross_term_name: 5 * fields_product(F2x[t1], F0x[t2], F0x[t3], self.k_configs) / self.num_triangles_per_bin / norm}
            self.logger.info('Computed quadrupole for cross term: %s', cross_term_name)
        
        return result

    @profile
    def compute_autoterms(self, F0x_species: np.ndarray, F2x_species: np.ndarray=None, norm: float=1.0) -> dict:
        """
        Computes auto-terms for a given species using vectorized operations. 
        This includes both monopole and quadrupole terms if quadrupole fields are available.

        Parameters
        ----------
        F0x_species : np.ndarray
            Fourier-space monopole field for the species.
        F2x_species : np.ndarray or None
            Fourier-space quadrupole field for the species.
        
        Returns
        -------
        dict
            Dictionary containing auto-term results for monopole and quadrupole.
        """
        
        auto_terms = {
            'monopole': fields_product(F0x_species, F0x_species, F0x_species, self.k_configs) / self.num_triangles_per_bin / norm
        }

        if F2x_species is not None:
            auto_terms['quadrupole'] = 5 * fields_product(F2x_species, F0x_species, F0x_species, self.k_configs) / self.num_triangles_per_bin / norm
        
        return auto_terms

    @profile
    def process_pair(self, Fx_files: dict, species: List[str]):
        """
        Processes a pair of species to compute cross-terms, which involve combinations of species 
        fields to calculate monopole and quadrupole terms. The valid combinations are filtered to avoid 
        same-species terms.

        Parameters
        ----------
        Fx_files : dict
            Dictionary containing Fourier-space fields for each species and pole.
        species : List[str]
            List containing the names of the two species in the pair.
        
        Returns
        -------
        tuple
            Tuple containing the species pair name and dictionary of cross-term results.
        """
        sp1 = self.map_name[species[0]] # Species 1
        sp2 = self.map_name[species[1]] # Species 2

        self.logger.info(f"Species 0: {species[0]}, Species 1: {species[1]}")

        # Load the fields for the species pair
        F0x_species1 = Fx_files[f"{self.F_poles[0]}_{species[0]}"] 
        F0x_species2 = Fx_files[f"{self.F_poles[0]}_{species[1]}"]

        if len(self.F_poles) > 1:
            F2x_species1 = Fx_files[f"{self.F_poles[1]}_{species[0]}"]
            F2x_species2 = Fx_files[f"{self.F_poles[1]}_{species[1]}"]
        else:
            F2x_species1 = F2x_species2 = None

        # Generate cross term names (e.g., tti, ttn, etc.)
        cross_term_names = sorted(set(''.join(tup) for tup in itertools.product(f'{sp1}{sp2}', repeat=3)))
        cross_term_names.remove(f'{sp1}{sp1}{sp1}')  # Remove same species terms
        cross_term_names.remove(f'{sp2}{sp2}{sp2}')  # Remove same species terms

        # List of combinations to process (the 6 valid ones)
        combs = [(t1, t2, t3) for t1, t2, t3 in itertools.product([0, 1], repeat=3) if (t1, t2, t3) not in [(0, 0, 0), (1, 1, 1)]]

        cross_term_values = {'monopole': {}, 'quadrupole': {}}
        
        for i, (t1, t2, t3) in enumerate(combs):
            
            prefactor = self.f_c[f'fc.{species[t1]}'] * self.f_c[f'fc.{species[t2]}'] * self.f_c[f'fc.{species[t3]}']
            self.prefacts[f'prefact.{cross_term_names[i]}'] = prefactor

            norm = np.power(self.I33[species[t1]] * self.I33[species[t2]] * self.I33[species[t3]], 1/3)

            result = self.compute_single_combination(t1, t2, t3, [F0x_species1, F0x_species2], 
                                                [F2x_species1, F2x_species2], norm, cross_term_names[i])
            
            cross_term_values['monopole'].update(result['monopole'])
            
            if result['quadrupole'] is not None:
                cross_term_values['quadrupole'].update(result['quadrupole'])
                    
        del F0x_species1, F0x_species2, F2x_species1, F2x_species2
        del result
        gc.collect()

        self.logger.info(f'Computed cross terms for species pair: {species[0]} & {species[1]}')

        return f'{sp1}_{sp2}', cross_term_values

    @profile
    def compute(self, Fx_files: dict, file_save_path: str=None, workers:int=None) -> dict:
        """
        Computes bispectrum terms for all species combinations, including auto- and cross-terms, 
        using parallel computation for efficiency. Optionally saves the computed terms to a file using pickle.

        Parameters
        ----------
        Fx_files : dict
            Dictionary containing Fourier-space fields for each species and pole.
        file_save_path : str, optional
            Path to save computed bispectrum terms; if None, results are not saved.
        workers : int, optional
            Number of workers for parallel computation; defaults to CPU count.
        
        Returns
        -------
        dict
            Dictionary containing all computed bispectrum terms for auto- and cross-species.
        """

        if workers is None:
            workers = os.cpu_count()

        nb.set_num_threads(workers)

        self.logger.info(f"Using {workers} workers for parallel computation.")

        self.logger.info("Starting cross term computation.")

        all_terms = {}

        # Iterate over each species pair and compute their cross terms
        for species_pair in tqdm(self.species_combs, total=len(self.species_combs), desc="Processing species pairs"):
            self.logger.info(f"Processing species pair: {species_pair[0]} & {species_pair[1]}")
            
            # Process each pair sequentially to limit the memory usage
            species_pair_name, cross_terms = self.process_pair(Fx_files, species_pair)
            all_terms[species_pair_name] = cross_terms
        
        self.logger.info("Computed all cross terms successfully.")


        # Iterate over each species to compute the auto-terms
        for species in tqdm(self.species_names):
            self.logger.info(f"Processing auto terms for species: {species}")

            F0x_species = Fx_files[f"{self.F_poles[0]}_{species}"]
            F2x_species = Fx_files[f"{self.F_poles[1]}_{species}"] if len(self.F_poles) > 1 else None

            prefactor = self.f_c[f'fc.{species}'] ** 3
            self.prefacts[f'prefact.{species}'] = prefactor

            # Compute Auto Terms 
            auto_terms = self.compute_autoterms(F0x_species, F2x_species, self.I33[species])
            self.logger.info("Computed auto terms for " + species + ".")

            # Store the auto-terms in the cross-terms dictionary as well (since it's part of the final output)
            all_terms[f'{self.map_name[species]}_{self.map_name[species]}'] = auto_terms

        self.logger.info("Computed all auto terms successfully.")


        if file_save_path is not None:
            try:
                with open(file_save_path, 'wb') as f:
                    pickle.dump(all_terms, f)
                self.logger.info("All bispectrum terms saved successfully.")
            except IOError as e:
                self.logger.error(f"Failed to save file at {file_save_path}: {e}")
                raise e
        return all_terms

class PowerSpectrumTerms:
    """
    A class to calculate and manage power spectrum terms (auto and cross) for species in a catalog.

    This class computes power spectrum terms (monopole and quadrupole) for both individual species (auto-terms) 
    and combinations of species (cross-terms), as well as the necessary fractional corrections based on provided I22 values. 
    It supports spherical averaging of fields within k-bins and allows the importation of precomputed field data from 
    specified directories.

    Attributes:
    -----------
    species_names : List[str]
        List of species names in the catalog.
        
    k_bin_indices : dict
        Dictionary mapping each k-bin to its respective indices in the field, used for spherical averaging.
        
    I22 : dict
        Dictionary containing I22 normalization values for each species, including a key 'total' for the combined catalog.
        
    logger : logging.Logger
        Logger used for tracking information, errors, and warnings during execution.
        
    species_combs : List[tuple]
        List of species pairs for cross-term calculation. Created only if there is more than one species.
        
    map_name : dict
        Dictionary mapping each species to a short identifier (typically the first letter of each species).
        
    F_poles : List[str]
        List of poles (e.g., 'F0_k', 'F2_k') to compute in the power spectrum. By default, it includes 'F0_k', 
        and 'F2_k' is added if specified.
        
    f_c : dict
        Dictionary of fraction values for each species relative to the combined catalog, based on the I22 normalization values.
    """
    def __init__(self, species_names: List[str], k_bin_indices: dict, poles: List[int], I22: dict, N0: dict) -> None:
        """
        Initializes PowerSpectrumTerms with species names, k-bin indices, poles, and normalization values.

        Parameters:
        -----------
        species_names : List[str]
            List of species names to analyze.
            
        k_bin_indices : dict
            A dictionary mapping each k-bin to field indices for spherical averaging.
            
        poles : List[int]
            List of integer poles to compute (e.g., monopole and quadrupole).
            
        I22 : dict
            Dictionary of I22 normalization values for each species and a 'total' key for the full catalog.

        N0 : dict
            Dictionary of shot noise values for each species.

        Raises:
        -------
        AssertionError
            If an I22 value is missing for any species or for the combined catalog ('total').
            
        ValueError
            If the first letter of species names is not unique.
        """
        self.species_names = species_names
        self.k_bin_indices = k_bin_indices
        self.I22 = I22
        self.N0 = N0

        assert all(species in list(I22.keys()) for species in species_names) == True, "I22 values not provided for all species."
        assert 'total' in I22.keys(), "I22 value not provided for the combined catalog. Expected key: 'total'."

        assert all(species in list(N0.keys()) for species in species_names) == True, "N0 values not provided for all species."
        
        self.logger = logging.getLogger(self.__class__.__name__)

        # Convert species names to lowercase
        self.species_names = list(map(str.lower, self.species_names))

        # Check if the first letter of the species name is unique
        if len(set([name[0] for name in self.species_names])) != len(self.species_names):
            self.logger.error("The first letter of the species name is not unique. Choose different names.")
            raise ValueError("The first letter of the species name is not unique.")

        # Generate combinations of species
        if len(self.species_names) > 1:
            self.species_combs = list(itertools.combinations(self.species_names, 2))
            self.species_combs = [sorted(species) for species in self.species_combs]
        else:
            self.logger.warning("Only auto-terms will be calculated since there is only one species.")
        self.logger.info("Species combinations: %s", self.species_combs)

        # Mapping file name suffix to a short species name
        self.map_name = {name: name[0].lower() for name in self.species_names}
        self.logger.info("Name mapping: %s", self.map_name)

        # Define poles to compute
        self.F_poles = ['F0_k']
        if 2 in poles:
            self.F_poles.append('F2_k')
        self.logger.info("Poles to compute: %s", self.F_poles)

        self.f_c = self.compute_species_fraction(self.I22, self.species_names)
        self.prefacts = {}
    
    def import_fields(self, Fk_save_dir: str, workers: int=None):
        """
        Imports precomputed fields for each species and pole from a specified directory.

        This method loads field data from a directory, using multiple workers for parallel file loading.

        Parameters:
        -----------
        Fk_save_dir : str
            Directory path where precomputed field files (e.g., F0_k, F2_k) are stored.
            
        workers : int, optional
            Number of workers to use for parallel loading of field files. Defaults to the number of CPU cores available.

        Returns:
        --------
        dict
            A dictionary mapping species and pole identifiers to the loaded field data (e.g., F0_k, F2_k).

        Raises:
        -------
        FileNotFoundError
            If the specified directory does not exist or is inaccessible.
        """

        self.logger.info("Loading files...")

        if workers is None:
            workers = os.cpu_count()
        
        if os.path.exists(Fk_save_dir) is False:
            raise FileNotFoundError(f"Directory {Fk_save_dir} does not exist.")

        # Load species files
        def load_file(file_path):
            """Helper function to load a single file."""
            return np.load(file_path, allow_pickle=False)
        
        filepaths = [os.path.join(Fk_save_dir, f"{F_pole}_{name}.npy") for F_pole in self.F_poles for name in self.species_names]

        pool_workers = min(workers, len(filepaths)) # Number of workers for parallel loading

        if pool_workers != workers:
            self.logger.info(f"Chose {pool_workers} workers for parallel loading to avoid thread contention.") 
        else:
            self.logger.info(f"Using {pool_workers} workers/threads for parallel loading.")

        # Parallel loading using ThreadPoolExecutor (multithreading)
        with ThreadPoolExecutor(max_workers=pool_workers) as executor:
            results = list(executor.map(load_file, filepaths))

        self.logger.info(f"{len(results)} files read successfully.")
        species_files = {f"{F_pole}_{name}": results[i] for i, (F_pole, name) in enumerate(itertools.product(self.F_poles, self.species_names))}
        self.logger.info("All files successfully loaded in a dictionary.")

        return species_files
    
    @profile
    def compute_species_fraction(self, I22: dict, species_names: List[str]) -> dict:
        """
        Computes the fraction for each species relative to the full catalog.

        This method calculates the fraction for each species using the provided I22 normalization values. 
        It normalizes by the I22 value of the combined catalog ('total').

        Parameters:
        -----------
        I22 : dict
            Dictionary of I22 normalization values for each species, including a key 'total' for the full catalog.
            
        species_names : List[str]
            List of species names to compute the fractions for.

        Returns:
        --------
        dict
            A dictionary mapping species names to their corresponding fractional value based on I22.
        """
        f_c = {}
        for species in species_names:
            f_c[f'fc.{species}'] = np.sqrt(I22[species] / I22['total'])
        return f_c

    @profile
    def spherical_average(self, field: np.ndarray) -> np.ndarray:
        """
        Computes the spherical average of a field within each k-bin.

        This method averages the input field over specified k-bins defined by the `k_bin_indices` attribute.

        Parameters:
        -----------
        field : np.ndarray
            Input field to average over the defined k-bins.

        Returns:
        --------
        np.ndarray
            Array of spherical averages, one for each k-bin.
        """
        spherical_averages = []
        for k_bin, indices in self.k_bin_indices.items():
            if (len(indices[0]) > 0): 
                # Compute the average of the values for specific k-bin
                spherical_averages.append(np.mean(field[indices]))
            else:
                # If there are no indices in this bin, set the average to NaN
                spherical_averages.append(np.nan)
        return np.array(spherical_averages).real
    
    @profile
    def compute_crossterms(self, Fk_files: dict, species_pair: List[str]) -> Tuple[str, dict]:
        """
        Computes the cross terms (monopole and quadrupole) between two species.

        This method calculates the cross terms for the monopole and quadrupole using the provided field data 
        for the species in the pair. 

        Parameters:
        -----------
        Fk_files : dict
            Dictionary containing the precomputed field files for each species and pole (e.g., F0_k, F2_k).
            
        species_pair : List[str]
            List of two species names for which the cross terms are to be computed.

        Returns:
        --------
        tuple
            A tuple containing the species pair name (e.g., 'species1_species2') and a dictionary with the computed 
            cross terms (monopole and quadrupole).
        """

        species1 = species_pair[0]
        species2 = species_pair[1]

        sp1 = self.map_name[species1] # Species 1
        sp2 = self.map_name[species2] # Species 2

        self.logger.info(f"Species 1: {species1}, Species 2: {species2}")

        # Load the fields for the species pair
        F0k_species1 = Fk_files[f"{self.F_poles[0]}_{species1}"] 
        F0k_species2 = Fk_files[f"{self.F_poles[0]}_{species2}"]

        if len(self.F_poles) > 1:
            F2k_species1 = Fk_files[f"{self.F_poles[1]}_{species1}"]
            F2k_species2 = Fk_files[f"{self.F_poles[1]}_{species2}"]
        else:
            F2k_species1 = F2k_species2 = None
    
        # Generate cross term names
        cross_term_names = [f'{sp1}{sp2}', f'{sp2}{sp1}']

        result = {'monopole': {}, 'quadrupole': {}}

        self.logger.info('Computing cross term: %s', cross_term_names)

        prefactor = self.f_c[f'fc.{species1}'] * self.f_c[f'fc.{species2}']

        self.prefacts[f'prefact.{sp1}_{sp2}'] = prefactor

        self.logger.info(f'Species fraction for {species1}: {self.f_c[f"fc.{species1}"]}')
        self.logger.info(f'Species fraction for {species2}: {self.f_c[f"fc.{species2}"]}')

        self.logger.info(f'Prefactor: {prefactor}')

        result['monopole'][f'{sp1}{sp2}'] = self.spherical_average(F0k_species1 * np.conjugate(F0k_species2)) / (np.sqrt(self.I22[species1] * self.I22[species2]))
        self.logger.info(f'Computed monopole for cross term: {sp1}{sp2}')

        result['monopole'][f'{sp2}{sp1}'] = self.spherical_average(F0k_species2 * np.conjugate(F0k_species1)) / (np.sqrt(self.I22[species1] * self.I22[species2]))
        self.logger.info(f'Computed monopole for cross term: {sp2}{sp1}')
        
        # If F2k fields are provided, compute quadrupole as well
        if F2k_species1 is not None and F2k_species2 is not None:
            result['quadrupole'][f'{sp1}{sp2}'] = 5 * self.spherical_average(F2k_species1 * np.conjugate(F0k_species2)) / (np.sqrt(self.I22[species1] * self.I22[species2]))
            self.logger.info(f'Computed quadrupole for cross term: {sp1}{sp2}')

            result['quadrupole'][f'{sp2}{sp1}'] = 5 * self.spherical_average(F2k_species2 * np.conjugate(F0k_species1)) / (np.sqrt(self.I22[species1] * self.I22[species2]))
            self.logger.info(f'Computed quadrupole for cross term: {sp2}{sp1}')

        del F0k_species1, F0k_species2, F2k_species1, F2k_species2
        gc.collect()

        self.logger.info(f'Computed cross terms for species pair: {sp1} & {sp2}')

        return f'{sp1}_{sp2}', result
    
    @profile
    def compute_autoterms(self, F0_k: np.ndarray, F2_k: np.ndarray, species_name: str) -> dict:
        """
        Computes the auto terms (monopole and quadrupole) for a given species.

        This method calculates the monopole and quadrupole terms for a species using the provided field data. 

        Parameters:
        -----------
        F0_k : np.ndarray
            The monopole field for the species.
            
        F2_k : np.ndarray or None
            The quadrupole field for the species. If not available, this can be set to `None`.
            
        species_name : str
            The species name for which to compute the auto terms.

        Returns:
        --------
        dict
            A dictionary containing the computed auto terms (monopole and quadrupole).
        """

        F0k_squared = np.abs(F0_k)**2
        F2k_squared = np.conjugate(F0_k) * F2_k if F2_k is not None else None

        prefactor = self.f_c[f'fc.{species_name}'] ** 2
        self.prefacts[f'prefact.{species_name}'] = prefactor

        auto_terms = {
            'monopole': self.spherical_average(F0k_squared) / self.I22[species_name] - self.N0[species_name],
            'quadrupole': 5 * self.spherical_average(F2k_squared) / self.I22[species_name] if F2_k is not None else None
        }

        return auto_terms

    @profile
    def compute(self, Fk_files: dict, file_save_path: str=None)-> dict:
        """
        Computes all cross and auto terms for the species in the catalog and stores them in a dictionary.

        This method computes the cross terms for species pairs and the auto terms for individual species, 
        and optionally saves the computed terms to a file.

        Parameters:
        -----------
        Fk_files : dict
            Dictionary containing the precomputed field data for each species and pole.
            
        file_save_path : str, optional
            Path to save the computed terms as a pickle file. If not provided, the terms are not saved.

        Returns:
        --------
        dict
            A dictionary containing the computed cross and auto terms for all species, along with the k-bin information.

        Raises:
        -------
        Exception
            If an error occurs while saving the computed terms to a file.
        """

        self.logger.info("Starting cross term computation.")

        all_terms = {}

        # Iterate over each species pair and compute their cross terms
        for species_pair in tqdm(self.species_combs, total=len(self.species_combs), desc="Processing species pairs"):
            self.logger.info(f"Processing species pair: {species_pair[0]} & {species_pair[1]}")
            
            # Process each pair sequentially to limit the memory usage
            species_pair_name, cross_terms = self.compute_crossterms(Fk_files, species_pair)
            all_terms[species_pair_name] = cross_terms
        
        self.logger.info("Computed all cross terms successfully.")

        # Iterate over each species to compute the auto-terms
        for species in tqdm(self.species_names):
            self.logger.info(f"Processing auto terms for species: {species}")

            F0k_species = Fk_files[f"{self.F_poles[0]}_{species}"]
            F2k_species = Fk_files[f"{self.F_poles[1]}_{species}"] if len(self.F_poles) > 1 else None
            
            # Compute Auto Terms 
            auto_terms = self.compute_autoterms(F0k_species, F2k_species, species)
            self.logger.info("Computed auto terms for " + species + ".")
            all_terms[f'{species}'] = auto_terms
        
        self.logger.info("Computed all auto terms successfully.")

        # Add k-bins to the dictionary
        all_terms['k_bins'] = list(self.k_bin_indices.keys())

        if file_save_path is not None:
            try:
                # Save the computed terms to a file
                with open(file_save_path, 'wb') as f:
                    pickle.dump(all_terms, f)
                self.logger.info("All power spectrum terms saved successfully to %s.", file_save_path)
            
            except Exception as e:
                self.logger.error(f"An error occurred: {e}")
                raise e

        return all_terms

#-------------------------------------------- END OF THE PIPELINE ------------------------------------------------#
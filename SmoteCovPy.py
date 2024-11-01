"""Importación de módulos"""
import pandas as pd
import numpy as np
from numpy import random as np_random
import random

"""Definición de una excepción personalizada"""
# exceptions
class WrongParamValueException(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)


"""Definición de constantes"""
# constants
SPACES = 100

class ShrinkageMethods:
    SKLEARN = "sklearn"
    WLM1KE = "WLM1ke"
    
class AttributeStats:
    MIN_MAX = 'min_max'
    MEAN = 'mean'
    STD = 'std'
    ALL = 'all'

"""Definición de la Clase Principal SmoteCov
Constructor de la clase SmoteCov"""  
# main class
class SmoteCov:
    """
    @description: SmoteCov is a module to balance an unbalanced dataset composed of binary classes and
    continuous attributes using the oversampling technique and exploiting the dependency relationships
    between class attributes using the covariance matrix
    """

    def __init__(self, filename: str, class_column_name: str = "Class"):
        self._filename = filename
        self._class_column_name = class_column_name
        self._dataset = pd.read_csv(filename)
        #
        self._minority_class_name = ""
        self._minority_class_count = 0
        self._majority_class_count = 0
        #
        self.get_classes_count()
        #
        self._covariance_matrix = None

    """Método para calcular el valor del Imbalance Ratio (IR)"""
    def calculate_ir_value(self, minority_class_plus: int = 0) -> float:
        """
        Returns Imbalance Ratio value
        :param minority_class_plus: adds synthetic instances array length to minority class length
        :return:
        """
        # calcular IR= len(clase mayoritaria) / len(clase minoritaria)
        ir = self._majority_class_count / (self._minority_class_count + minority_class_plus)
        # log
        # print(f"[OK] Imbalance Ratio: {ir}")
        return ir

    """Método para contar las instancias de cada clase y determinar las clases mayoritaria y minoritaria"""
    def get_classes_count(self) -> tuple:
        """
        Sets minority and majority class info
        :return: (minority_class_count, majority_class_count)
        """
        # contar cantidad de instancias de cada clase con NumPy
        values_counts = self._dataset[self._class_column_name].value_counts()
        # obtener la lista de nombres de clases
        classes = values_counts.index.tolist()
        # nombre de la clase minoritaria
        self._minority_class_name = classes[-1]
        # obtener valores del conteo
        self._majority_class_count = values_counts[0]
        self._minority_class_count = values_counts[1]
        # log
        print("=" * SPACES)
        print(f"[OK] Minority classname: {self._minority_class_name}")
        print(f"[OK] Minority class instances count: {self._minority_class_count}")
        print(f"[OK] Majority class instances count: {self._majority_class_count}")
        print("=" * SPACES)
        return self._minority_class_count, self._majority_class_count

    """Método para obtener el grupo de la clase minoritaria"""
    def _get_minority_class_group(self) -> pd.DataFrame:
        #
        group_by_classname = self._dataset.groupby(self._class_column_name)
        group_by_classname = group_by_classname.get_group(self._minority_class_name)
        return group_by_classname.drop(self._class_column_name, axis=1)

    """Método para obtener la matriz de covarianza clásica """
    def _get_classic_cov_matrix(self) -> None:
        """
        Determines the covariance matrix for minority class matrix
        :return:
        """
        # obtener matriz de covarianza
        cov_matrix = self._get_minority_class_group().cov()
        self._covariance_matrix = np.asmatrix(cov_matrix)

    """Método para obtener la matriz de covarianza usando Ledoit-Wolf"""
    def _get_ledoit_wolf_cov_matrix(self, shrinkage_method:str) -> None:
        """
        Determines the cov matrix for minority class instances by using Ledoit-Wolf
        :param shrinkage_method: use any value from ShrinkageMethods class
        :return:
        """
        # obtener matriz de covarianza
        min_class_matrix = self._get_minority_class_group()
        ledoit_wolf = None
        if shrinkage_method == ShrinkageMethods.SKLEARN:
            from sklearn.covariance import LedoitWolf
            ledoit_wolf = LedoitWolf()
            ledoit_wolf.fit(min_class_matrix)
        elif shrinkage_method == ShrinkageMethods.WLM1KE:
            from ledoit_wolf import LedoitWolf
            ledoit_wolf = LedoitWolf()
            ledoit_wolf.shrinkage(np.array(min_class_matrix))
        else:
            raise WrongParamValueException(f"Unknow parameter: '{shrinkage_method}'. Use: ShrinkageMethods class's constants instead")
        self._covariance_matrix = np.asmatrix(ledoit_wolf.covariance_)

    """Método para obtener estadísticas de los atributos """
    def _get_attributes_stats(self, stat:str=AttributeStats.ALL) -> np.ndarray:
        """
        Returns an array with statistics values
        :param stat: use any value from AttributeStats class
        :return:
        """
        # encontrar valores minimos, maximos, media y desviacion estandar para cada atributo
        # array con nombre de las columnas
        columns_name = self._dataset.columns.values
        # array con valores estadisticos
        attributes_stats_array = np.array([])
        data = self._covariance_matrix
        for i in range(np.shape(data)[1]):
            name_value = columns_name[i]
            min_value = data[:, i].min()
            max_value = data[:, i].max()
            mean_value = data[:, i].mean()
            std_value = data[:, i].std()
            if stat == AttributeStats.ALL:
                #
                attributes_stats_array = np.append(attributes_stats_array,
                                                [{'name':name_value, 'min':min_value, 'max':max_value, 'mean':mean_value,
                                                                'std':std_value}],
                                                axis=0)
            elif stat == AttributeStats.MEAN:
                attributes_stats_array = np.append(attributes_stats_array, [mean_value], axis=0)
            elif stat == AttributeStats.MIN_MAX:
                attributes_stats_array = np.append(attributes_stats_array, [{'min':min_value, 'max':max_value}], axis=0)
            else:
                raise WrongParamValueException(f"Unknow parameter: '{stat}'. Use: AttributeStats class's constants instead")
        return attributes_stats_array

    """Método para balancear el dataset usando el método clásico """
    def balance_dataset_using_classic_method(self) -> None:
        """
        Generates 2 CSV files:
        - 'In range balanced dataset.csv' file
        - 'Free range balanced dataset.csv' file
        Both files contains synthetic instances created by using classic covariance matrix
        """
        print("Balance dataset by using CLASSIC method")
        print("*"*SPACES)
        #
        in_range_balanced_dataset = None
        free_range_balanced_dataset = None
        if self.calculate_ir_value() > 1.5:
            # obtener la matriz de covarianza
            self._get_classic_cov_matrix()
            # obtener las estadisticas para cada variable de la instancia
            attributes_array = self._get_attributes_stats()
            #
            while self.calculate_ir_value(
                    np.shape(in_range_balanced_dataset)[0] if in_range_balanced_dataset is not None else 0) > 1.5:
                in_range_instance = []
                free_range_instance = []
                # 
                for i in range(len(attributes_array)):
                    attribute: dict = attributes_array[i]
                    # se genera un valor float al azar dentro del rango [attributeMin, attributeMax]
                    in_range_instance.append(random.uniform(attribute['min'], attribute['max']))
                    # se genera un valor float al azar a partir de la distribucion gaussiana
                    free_range_value = random.gauss(attribute['mean'], attribute['std'])
                    # se asegura de que todos los valores generados al azar esten fuera del rango
                    # **** actualmente NO SE USA ****
                    #while free_range_value >= attribute['min'] and free_range_value <= attribute['max']:
                    #    free_range_value = random.gauss(attribute['mean'], attribute['std'])
                    free_range_instance.append(free_range_value)
                    #
                # insertar instancias en su respectiva matriz
                if in_range_balanced_dataset is None or free_range_balanced_dataset is None:
                    in_range_balanced_dataset = np.matrix([in_range_instance])
                    free_range_balanced_dataset = np.matrix([free_range_instance])
                else:
                    in_range_balanced_dataset = np.append(in_range_balanced_dataset, [in_range_instance], axis=0)
                    free_range_balanced_dataset = np.append(free_range_balanced_dataset, [free_range_instance],
                                                              axis=0)
            #
            print("[OK] Instances array was generated successfully")
            # guardar el dataset balanceado en un archivo
            print("=" * SPACES)
            print("[-] Generating CSV files...")
            self.save_to_csv_file("in_range_balanced_dataset_classic_method.csv", in_range_balanced_dataset)
            self.save_to_csv_file("free_range_balanced_dataset_classic_method.csv", free_range_balanced_dataset)
        else:
            print(
                f"[OK] The dataset is balanced (IR: {self.calculate_ir_value()} | min={self._minority_class_count}, maj={self._majority_class_count})")
        print("=" * SPACES)

    """Método para balancear el dataset usando el método Ledoit-Wolf"""
    def balance_dataset_using_ledoit_wolf_method(self, shrinkage_method:str) -> None:
        """
        Generates 2 CSV files:
        - 'In range balanced dataset.csv' file
        - 'Free range balanced dataset.csv' file
        Both files contains synthetic instances created by using Ledoit-Wolf covariance matrix with dependent values
        """
        print(f"Balance dataset by using LEDOIT-WOLF [{shrinkage_method}] method")
        print("*"*SPACES)
        #
        def fit_to_range(instance:np.ndarray, min_max_array:np.ndarray, mean_array:np.ndarray, where:str="in") -> np.ndarray:
            """
            Adjust instance's variables values
            :param instance: a row from minority class matrix
            :param min_max_array: 
            :param mean_array: 
            :param where: 'in' or 'out'
            :return:
            """
            if where=="in":
                # se asegura de que todos los valores para cada variable esten dentro del rango
                for i in range(len(instance)):
                    attribute_value = instance[i]
                    if attribute_value > min_max_array[i]['max']:
                        instance[i] = min_max_array[i]['max']
                    elif attribute_value < min_max_array[i]['min']:
                        instance[i] = min_max_array[i]['min']
            elif where=="out":
                # se asegura de que todos los valores para cada variable esten fuera del rango
                # **** actualmente NO SE USA ****
                for i in range(len(instance)):
                    attribute_value = instance[i]
                    while attribute_value <= min_max_array[i]['max'] and attribute_value >= min_max_array[i]['min']:
                        new_instance = np_random.multivariate_normal(mean_array, self._covariance_matrix)
                        attribute_value = new_instance[i]
                    if attribute_value != instance[i]:
                        instance[i] = attribute_value
            else:
                raise WrongParamValueException(f"Unknow parameter: '{where}'. Use: 'in', 'out' instead")
            return instance
        #
        in_range_balanced_dataset = None
        free_range_balanced_dataset = None
        if self.calculate_ir_value() > 1.5:
            # obtener la matriz de covarianza con Ledoit-Wolf
            self._get_ledoit_wolf_cov_matrix(shrinkage_method)
            # array de medias para cada variable
            mean_array  = self._get_attributes_stats(AttributeStats.MEAN)
            # array de minimos y maximos para cada variable
            min_max_array  = self._get_attributes_stats(AttributeStats.MIN_MAX)
            while self.calculate_ir_value(
                    np.shape(in_range_balanced_dataset)[0] if in_range_balanced_dataset is not None else 0) > 1.5:
                # generar la instancia
                instance:np.ndarray = np_random.multivariate_normal(mean=mean_array, cov=self._covariance_matrix)
                # ajustar valores de las variables de la instancia para 'dentro del rango'
                in_range_instance = fit_to_range(instance, min_max_array, mean_array, "in")
                # para 'fuera del rango' se toma la instancia tal cual
                free_range_instance = instance # fit_to_range(instance, min_max_array, mean_array, "out")
                # insertar en su respectiva matriz
                if in_range_balanced_dataset is None or free_range_balanced_dataset is None:
                    in_range_balanced_dataset = np.matrix([in_range_instance])
                    free_range_balanced_dataset = np.matrix([free_range_instance])
                else:
                    in_range_balanced_dataset = np.append(in_range_balanced_dataset, [in_range_instance], axis=0)
                    free_range_balanced_dataset = np.append(free_range_balanced_dataset, [free_range_instance], axis=0)
            #
            print("[OK] Instances array was generated successfully!")
            # guardar el dataset balanceado en un archivo
            print("=" * SPACES)
            print("[-] Generating CSV files...")
            self.save_to_csv_file(f"in_range_balanced_dataset_ld_method_by_{shrinkage_method.lower()}.csv", in_range_balanced_dataset)
            self.save_to_csv_file(f"free_range_balanced_dataset_ld_method_by_{shrinkage_method.lower()}.csv", free_range_balanced_dataset)
        else:
            print(
                f"[OK] The dataset is balanced (IR: {self.calculate_ir_value()} | min={self._minority_class_count}, maj={self._majority_class_count})")
        print("=" * SPACES)

    def save_to_csv_file(self, filename: str, sythetic_instances_dataset: np.matrix) -> None:
        """
        Saves the balanced dataset in a CSV file
        :param filename: CSV file's name
        :param synthetic_instances_dataset: synthetic instances dataset
        :return:
        """
        try:
            try:
                filename = self._filename.split(".")[0] + "_" + filename
            except:
                pass
            imbalanced_dataset_str = open(self._filename, "r").read()
            finish_with_new_line = imbalanced_dataset_str.endswith("\n")
            with open(filename, "w") as file:
                str_dataset = imbalanced_dataset_str + ("\n" if not finish_with_new_line else "")
                rows, columns = np.shape(sythetic_instances_dataset)
                for i in range(rows):
                    for j in range(columns):
                        str_dataset += str(sythetic_instances_dataset[i, j]) + ", "
                    str_dataset += self._minority_class_name.strip()
                    str_dataset += "\n"
                file.write(str_dataset)
                print(f"[OK] File \"{filename}\" was saved successfuly!")
        except Exception as e:
            print(f"[ERROR] File \"{filename}\" could't be saved (details ↓)")
            print(e.args)
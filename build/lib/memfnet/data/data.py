import functools
import os
import sys
import numpy as np
import pandas as pd
import torch
from pymatgen.core.composition import Composition
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F

from memfnet.core import Featurizer

class BatteryData_(Dataset):
    """
    The CompositionData dataset is a wrapper for a dataset data points are
    automatically constructed from composition strings.
    """

    def __init__(
        self,
        data_path,
        fea_path,
        identifiers=["material_id", "composition"],
        add_noise = False,
    ):
        """[summary]

        Args:
            data_path (str): [description]
            fea_path (str): [description]
            task_dict ({name: task}): list of tasks
            identifiers (list, optional): column names for unique identifier
                and pretty name. Defaults to ["id", "composition"].
        """

        assert len(identifiers) == 2, "Two identifiers are required"

        self.identifiers = identifiers
        self.add_noise = add_noise

        assert os.path.exists(data_path), f"{data_path} does not exist!"
        # NOTE make sure to use dense datasets,
        # NOTE do not use default_na as "NaN" is a valid material

        data_list = []
        for filename in os.listdir(data_path):
            if '.csv' in filename:
                print(filename)
                df = pd.read_csv(os.path.join(data_path, filename), keep_default_na=False, na_values=[])
                data_list.append(df)

        self.df = pd.concat(data_list, axis=0, ignore_index=True)



        assert os.path.exists(fea_path), f"{fea_path} does not exist!"
        self.elem_features = Featurizer.from_json(fea_path)
        self.elem_emb_len = self.elem_features.embedding_size



    def __len__(self):
        return len(self.df)

    @functools.lru_cache(maxsize=None)  # Cache data for faster training
    def __getitem__(self, idx):
        """[summary]

        Args:
            idx (int): dataset index

        Raises:
            AssertionError: [description]
            ValueError: [description]

        Returns:
            atom_weights: torch.Tensor shape (M, 1)
                weights of atoms in the material
            atom_fea: torch.Tensor shape (M, n_fea)
                features of atoms in the material
            self_fea_idx: torch.Tensor shape (M*M, 1)
                list of self indices
            nbr_fea_idx: torch.Tensor shape (M*M, 1)
                list of neighbor indices
            target: torch.Tensor shape (1,)
                target value for material
            cry_id: torch.Tensor shape (1,)
                input id for the material

        """
        df_idx = self.df.iloc[idx]
        composition = df_idx['composition']   # 直接使用
        V_low = df_idx['V_low']
        V_high = df_idx['V_high']
        rate = df_idx['rate']
        cycle = df_idx['cycle']
        Vii = df_idx['Vii']
        sin1_temp = torch.tensor([df_idx['sin1_temp']], dtype=torch.float32)
        sin1_time = torch.tensor([df_idx['sin1_time']], dtype=torch.float32)
        sin2_temp = torch.tensor([df_idx['sin2_temp']], dtype=torch.float32)
        sin2_time = torch.tensor([df_idx['sin2_time']], dtype=torch.float32)
        sin2_exists = torch.tensor([df_idx['sin2_exists']], dtype=torch.float32)

        material_id = df_idx["material_id"]

        cry_ids = df_idx[self.identifiers].values
        comp_dict = Composition(composition).get_el_amt_dict()

        elements = list(comp_dict.keys())
        # print("elements: ", elements)
        num_nodes = len(elements)  # 节点数（元素数量）

        weights = list(comp_dict.values())
        weights = np.atleast_2d(weights).T / 2

        if self.add_noise:
            weights += np.random.rand(len(weights), 1) * 1e-3
            weights = np.clip(weights, 0, 1)

        try:
            # 直接获取每个元素的特征向量，无需考虑 F
            atom_fea = np.vstack(
                [self.elem_features.get_fea(element) for element in elements]
            )
        except AssertionError:
            raise AssertionError(
                f"cry-id {cry_ids[0]} [{composition}] contains element types not in embedding"
            )
        except ValueError:
            raise ValueError(
                f"cry-id {cry_ids[0]} [{composition}] composition cannot be parsed into elements"
            )
        # print("atom_fea shape:", atom_fea.shape)

        nele = len(elements)
        self_fea_idx = []
        nbr_fea_idx = []

        ###################################只有Li-O和TM-O########################################
        # 正向连接：Li 和 M -> O
        for i, element in enumerate(elements):
            if element == 'Li':  # Li 只与 O 连接
                nbr_indices = [j for j, ele in enumerate(elements) if ele == 'O']
                if nbr_indices:
                    nbr_fea_idx += nbr_indices
                    self_fea_idx += [i] * len(nbr_indices)
                
            elif element not in ['Li', 'O']:  # M 只与 O 连接
                nbr_indices = [j for j, ele in enumerate(elements) if ele == 'O']
                if nbr_indices:
                    nbr_fea_idx += nbr_indices
                    self_fea_idx += [i] * len(nbr_indices)

        # 反向连接：O -> Li 和 M
        o_index = elements.index('O')
        for i, element in enumerate(elements):
            if element != 'O':  # 只向 Li 和 M 连接
                self_fea_idx.append(o_index)  # O 作为源节点
                nbr_fea_idx.append(i)         # Li 或 M 作为目标节点

        # convert all data to tensors
        atom_weights = torch.tensor(weights, requires_grad = True, dtype= torch.float32) # (5,1)torch.Tensor(weights)
        atom_fea = torch.Tensor(atom_fea) # (num_elem, 200)
        self_fea_idx = torch.LongTensor(self_fea_idx)
        nbr_fea_idx = torch.LongTensor(nbr_fea_idx)
        V_window = torch.Tensor([V_low, V_high])
        rate = torch.Tensor([rate])
        cycle = torch.Tensor([cycle]) # (1, )
        Vii = torch.tensor([Vii], requires_grad = True, dtype= torch.float32)
        material_id = torch.Tensor([material_id]) # (1, )
        num_nodes = torch.Tensor([num_nodes]) # (1, )

        
        Qii = df_idx['Qii']
        dQdVii = df_idx['dQdVii']
        targets = [torch.Tensor([Qii]), torch.Tensor([dQdVii])]

        return (
            (atom_weights, atom_fea, self_fea_idx, nbr_fea_idx, V_window, rate, cycle, Vii, 
             sin1_temp, sin1_time,sin2_temp,sin2_time,sin2_exists, # 合成条件
             material_id, num_nodes,
             ),
            targets,
            *cry_ids,
        )



class read_battery_infor(Dataset):
    """
    Return the battery test information
    """

    def __init__(
        self,
        data_path,
        fea_path,
        identifiers=["material_id", "composition"],
        add_noise = False,
        # identifiers=["material_id", "composition"],
    ):
        """[summary]

        Args:
            data_path (str): [description]
            fea_path (str): [description]
            task_dict ({name: task}): list of tasks
            identifiers (list, optional): column names for unique identifier
                and pretty name. Defaults to ["id", "composition"].
        """

        assert len(identifiers) == 2, "Two identifiers are required"

        self.identifiers = identifiers
        self.add_noise = add_noise

        assert os.path.exists(data_path), f"{data_path} does not exist!"
        # NOTE make sure to use dense datasets,
        # NOTE do not use default_na as "NaN" is a valid material

        data_list = []
        for filename in os.listdir(data_path):
            if '.csv' in filename:
                print(filename)
                df = pd.read_csv(os.path.join(data_path, filename), keep_default_na=False, na_values=[])
                first_row = df.iloc[1]

                print(first_row)
                data_list.append(first_row)

        self.df = pd.concat(data_list, axis=0, ignore_index=True)



        assert os.path.exists(fea_path), f"{fea_path} does not exist!"
        self.elem_features = Featurizer.from_json(fea_path)
        self.elem_emb_len = self.elem_features.embedding_size



    def __len__(self):
        return len(self.df)

    def get_df(self):
        return self.df

    @functools.lru_cache(maxsize=None)  # Cache data for faster training
    def __getitem__(self, idx):
        """[summary]

        Args:
            idx (int): dataset index

        Raises:
            AssertionError: [description]
            ValueError: [description]

        Returns:
            atom_weights: torch.Tensor shape (M, 1)
                weights of atoms in the material
            atom_fea: torch.Tensor shape (M, n_fea)
                features of atoms in the material
            self_fea_idx: torch.Tensor shape (M*M, 1)
                list of self indices
            nbr_fea_idx: torch.Tensor shape (M*M, 1)
                list of neighbor indices
            target: torch.Tensor shape (1,)
                target value for material
            cry_id: torch.Tensor shape (1,)
                input id for the material

        """
        df_idx = self.df.iloc[idx]
        composition = df_idx[['composition']][0]
        V_low = df_idx[['V_low']][0]
        V_high =  df_idx[['V_high']][0]
        rate = df_idx[['rate']][0]
        cycle = df_idx[['cycle']][0]
        Vii = df_idx[['Vii']][0]

        cry_ids = df_idx[self.identifiers].values
        comp_dict = Composition(composition).get_el_amt_dict()


        if self.add_noise:
            F_content = (2.0 - comp_dict['O']) / 2 + np.random.rand(1) * 1e-3
            F_content = np.clip(F_content, 0, 1)
        else:
            F_content = (2.0 - comp_dict['O']) / 2

        try:
            comp_dict.pop('F')
            comp_dict.pop('O')
        except:
            comp_dict.pop('O')

        elements = list(comp_dict.keys())

        weights = list(comp_dict.values())
        weights = np.atleast_2d(weights).T / 2

        if self.add_noise:
            weights += np.random.rand(len(weights), 1) * 1e-3
            weights = np.clip(weights, 0, 1)

        try:
            atom_fea = np.vstack(
                [self.elem_features.get_fea(element) + self.elem_features.get_fea('F') * F_content for element in elements]
            )
        except AssertionError:
            raise AssertionError(
                f"cry-id {cry_ids[0]} [{composition}] contains element types not in embedding"
            )
        except ValueError:
            raise ValueError(
                f"cry-id {cry_ids[0]} [{composition}] composition cannot be parsed into elements"
            )

        nele = len(elements)
        self_fea_idx = []
        nbr_fea_idx = []

        for i, _ in enumerate(elements):
            self_fea_idx += [i] * nele
            nbr_fea_idx += list(range(nele))

        # convert all data to tensors
        atom_weights = torch.tensor(weights, requires_grad = True, dtype= torch.float32) # torch.Tensor(weights)
        atom_fea = torch.Tensor(atom_fea)
        self_fea_idx = torch.LongTensor(self_fea_idx)
        nbr_fea_idx = torch.LongTensor(nbr_fea_idx)
        V_window = torch.Tensor([V_low, V_high])
        rate = torch.Tensor([rate])
        cycle = torch.Tensor([cycle])
        Vii = torch.tensor([Vii], requires_grad = True, dtype= torch.float32)

        Q0ii = df_idx['Q0ii']
        Qii = df_idx['Qii']
        dQdVii = df_idx['dQdVii']

        targets = [torch.Tensor([Q0ii]), torch.Tensor([Qii]), torch.Tensor([dQdVii]) ]
        # for target in self.task_dict:
        #     if self.task_dict[target] == "regression":
        #         targets.append(torch.Tensor([df_idx[target]]))
        #     elif self.task_dict[target] == "classification":
        #         targets.append(torch.LongTensor([df_idx[target]]))

        return (
            (atom_weights, atom_fea, self_fea_idx, nbr_fea_idx, V_window, rate, cycle,  Vii),
            targets,
            *cry_ids,
        )

def expand_to_node_level(batch_material_id, batch_cycle, batch_num_nodes,
                         batch_sin1_temp, batch_sin1_time, batch_sin2_temp, batch_sin2_time, batch_vii):
    """
    将样本级别的 batch_material_id、batch_cycle 和其他特征扩展到节点级别。

    Args:
        batch_material_id: Tensor, shape (batch_size, 1), 每个样本的 material_id。
        batch_cycle: Tensor, shape (batch_size, 1), 每个样本的 cycle。
        batch_num_nodes: Tensor, shape (batch_size, 1), 每个样本的节点数。
        batch_sin1_temp: Tensor, shape (batch_size, 1), 每个样本的 sin1_temp。
        batch_sin1_time: Tensor, shape (batch_size, 1), 每个样本的 sin1_time。
        batch_sin2_temp: Tensor, shape (batch_size, 1), 每个样本的 sin2_temp。
        batch_sin2_time: Tensor, shape (batch_size, 1), 每个样本的 sin2_time。

    Returns:
        expanded_material_id: Tensor, shape (num_nodes_total, 1)
        expanded_cycle: Tensor, shape (num_nodes_total, 1)
        expanded_sin1_temp: Tensor, shape (num_nodes_total, 1)
        expanded_sin1_time: Tensor, shape (num_nodes_total, 1)
        expanded_sin2_temp: Tensor, shape (num_nodes_total, 1)
        expanded_sin2_time: Tensor, shape (num_nodes_total, 1)
    """
    expanded_material_id = []
    expanded_cycle = []
    expanded_vii = []
    expanded_sin1_temp = []
    expanded_sin1_time = []
    expanded_sin2_temp = []
    expanded_sin2_time = []

    for (
        material_id, 
        cycle, 
        num_nodes, 
        sin1_temp, 
        sin1_time, 
        sin2_temp, 
        sin2_time,
        vii
    ) in zip(
        batch_material_id, 
        batch_cycle, 
        batch_num_nodes,
        batch_sin1_temp,
        batch_sin1_time,
        batch_sin2_temp,
        batch_sin2_time,
        batch_vii
    ):
        expanded_material_id.extend([material_id.item()] * int(num_nodes.item()))
        expanded_cycle.extend([cycle.item()] * int(num_nodes.item()))
        expanded_sin1_temp.extend([sin1_temp.item()] * int(num_nodes.item()))
        expanded_sin1_time.extend([sin1_time.item()] * int(num_nodes.item()))
        expanded_sin2_temp.extend([sin2_temp.item()] * int(num_nodes.item()))
        expanded_sin2_time.extend([sin2_time.item()] * int(num_nodes.item()))
        expanded_vii.extend([vii.item()] * int(num_nodes.item()))

    # 转换为 Tensor
    expanded_material_id = torch.tensor(expanded_material_id, dtype=torch.long).unsqueeze(-1)
    expanded_cycle = torch.tensor(expanded_cycle, dtype=torch.float32).unsqueeze(-1)
    expanded_sin1_temp = torch.tensor(expanded_sin1_temp, dtype=torch.float32).unsqueeze(-1)
    expanded_sin1_time = torch.tensor(expanded_sin1_time, dtype=torch.float32).unsqueeze(-1)
    expanded_sin2_temp = torch.tensor(expanded_sin2_temp, dtype=torch.float32).unsqueeze(-1)
    expanded_sin2_time = torch.tensor(expanded_sin2_time, dtype=torch.float32).unsqueeze(-1)
    expanded_vii = torch.tensor(expanded_vii, dtype=torch.float32).unsqueeze(-1)

    return expanded_material_id, expanded_cycle, expanded_sin1_temp, expanded_sin1_time, expanded_sin2_temp, expanded_sin2_time, expanded_vii
    


def collate_batch_(dataset_list):
    """
    Collate a list of data and return a batch for predicting crystal
    properties.

    Parameters
    ----------

    dataset_list: list of tuples for each data point.
      (atom_fea, nbr_fea, nbr_fea_idx, target)

      atom_fea: torch.Tensor shape (n_i, atom_fea_len)
      nbr_fea: torch.Tensor shape (n_i, M, nbr_fea_len)
      self_fea_idx: torch.LongTensor shape (n_i, M)
      nbr_fea_idx: torch.LongTensor shape (n_i, M)
      target: torch.Tensor shape (1, )
      cif_id: str or int

    Returns
    -------
    N = sum(n_i); N0 = sum(i)

    batch_atom_weights: torch.Tensor shape (N, 1)
    batch_atom_fea: torch.Tensor shape (N, orig_atom_fea_len)
        Atom features from atom type
    batch_self_fea_idx: torch.LongTensor shape (N, M)
        Indices of mapping atom to copies of itself
    batch_nbr_fea_idx: torch.LongTensor shape (N, M)
        Indices of M neighbors of each atom
    crystal_atom_idx: list of torch.LongTensor of length N0
        Mapping from the crystal idx to atom idx
    target: torch.Tensor shape (N, 1)
        Target value for prediction
    batch_comps: list
    batch_ids: list
    """
    # define the lists
    batch_atom_weights = []
    batch_atom_fea = []
    batch_self_fea_idx = []
    batch_nbr_fea_idx = []
    batch_window = []
    batch_rate = []
    batch_cycle = []
    batch_Vii = []
    batch_material_id = []
    batch_num_nodes = []
    batch_sin1_temp = []
    batch_sin1_time = []
    batch_sin2_temp = []
    batch_sin2_time = []
    batch_sin2_exists = []

    crystal_atom_idx = []
    batch_targets = []
    batch_cry_ids = []

    cry_base_idx = 0
    for i, (inputs, target, *cry_ids) in enumerate(dataset_list):

        atom_weights, atom_fea, self_fea_idx, nbr_fea_idx, V_window, rate, cycle, Vii, sin1_temp, sin1_time, sin2_temp, sin2_time, sin2_exists, material_id, num_nodes = inputs

        # number of atoms for this crystal
        n_i = atom_fea.shape[0]

        # batch the features together
        batch_atom_weights.append(atom_weights)
        batch_atom_fea.append(atom_fea)
        batch_window.append(V_window)
        batch_rate.append(rate)
        batch_cycle.append(cycle)
        batch_Vii.append(Vii)
        batch_material_id.append(material_id)
        batch_num_nodes.append(num_nodes)
        batch_sin1_temp.append(torch.tensor(sin1_temp, dtype=torch.float32) if not isinstance(sin1_temp, torch.Tensor) else sin1_temp)
        batch_sin1_time.append(torch.tensor(sin1_time, dtype=torch.float32) if not isinstance(sin1_time, torch.Tensor) else sin1_time)
        batch_sin2_temp.append(torch.tensor(sin2_temp, dtype=torch.float32) if not isinstance(sin2_temp, torch.Tensor) else sin2_temp)
        batch_sin2_time.append(torch.tensor(sin2_time, dtype=torch.float32) if not isinstance(sin2_time, torch.Tensor) else sin2_time)
        batch_sin2_exists.append(torch.tensor(sin2_exists, dtype=torch.float32) if not isinstance(sin2_exists, torch.Tensor) else sin2_exists)

        # mappings from bonds to atoms
        batch_self_fea_idx.append(self_fea_idx + cry_base_idx)
        batch_nbr_fea_idx.append(nbr_fea_idx + cry_base_idx)

        # mapping from atoms to crystals
        crystal_atom_idx.append(torch.tensor([i] * n_i))

        # batch the targets and ids
        batch_targets.append(target)
        batch_cry_ids.append(cry_ids)

        # increment the id counter
        cry_base_idx += n_i

    batch_atom_weights = torch.cat(batch_atom_weights, dim=0)
    batch_atom_fea = torch.cat(batch_atom_fea, dim=0)
    batch_self_fea_idx = torch.cat(batch_self_fea_idx, dim=0)
    batch_nbr_fea_idx = torch.cat(batch_nbr_fea_idx, dim=0)
    crystal_atom_idx = torch.cat(crystal_atom_idx)
    batch_window = torch.stack(batch_window)
    batch_rate = torch.stack(batch_rate)
    batch_cycle = torch.stack(batch_cycle)
    batch_Vii = torch.stack(batch_Vii)
    batch_material_id = torch.stack(batch_material_id)
    batch_num_nodes = torch.stack(batch_num_nodes) # (bacth_size, 1)
    batch_sin1_temp = torch.stack(batch_sin1_temp)
    batch_sin1_time = torch.stack(batch_sin1_time)
    batch_sin2_temp = torch.stack(batch_sin2_temp)
    batch_sin2_time = torch.stack(batch_sin2_time)
    
    expand_material_id, expand_cycle, expand_sin1_temp, expand_sin1_time, expand_sin2_temp, expand_sin2_time, expand_vii = expand_to_node_level(batch_material_id, batch_cycle, batch_num_nodes, batch_sin1_temp, batch_sin1_time, batch_sin2_temp, batch_sin2_time, batch_Vii) # (bacth*elem_num, 1)
    return (
        (
            batch_atom_weights,
            batch_atom_fea,
            batch_self_fea_idx,
            batch_nbr_fea_idx,
            crystal_atom_idx,
            batch_window,
            batch_rate,
            batch_cycle,
            batch_Vii,
            expand_cycle,
            expand_sin1_temp,
            expand_sin1_time,
            expand_sin2_temp,
            expand_sin2_time,
            expand_vii
        ),
        tuple(torch.stack(b_target, dim=0) for b_target in zip(*batch_targets)),
        *zip(*batch_cry_ids),
    )

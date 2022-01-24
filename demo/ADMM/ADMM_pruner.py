import paddle
import numpy as np

class AdmmPruner():
    def __init__(self,
                program,
                ratio=0.55,
                scope=None,
                place=None):
        self.ratio = ratio
        self.scope = paddle.static.global_scope() if scope is None else scope         
        self.place = paddle.static.cpu_places()[0] if place is None else place 
        self.masks = self._apply_masks(program)
    
    def _apply_masks(self, program):
        params = []
        masks = []
        self.no_grad_set = set()
        for param in program.all_parameters():
            if param.name.split('_')[-1] != 'weights':
                continue
            if not (len(param.shape) == 4 and param.shape[2] == 1 
                    and param.shape[3] == 1):
                continue
            # # create mask
            mask = program.global_block().create_var(
                name=param.name + "_mask",
                shape=param.shape,
                dtype=param.dtype,
                type=param.type,
                persistable=param.persistable,
                stop_gradient=True)
            self.scope.var(mask.name).get_tensor().set(np.ones(mask.shape).astype("float32"), self.place)

            params.append(param)
            masks.append(mask)
            self.no_grad_set.add(mask.name)
                
            with paddle.static.program_guard(main_program=program):
                block = program.global_block()
                for param, mask in zip(params, masks):
                    block._prepend_op(
                        type='elementwise_mul',
                        inputs={'X': param,
                                'Y': mask},
                        outputs={'Out': param},
                        attrs={'axis': -1,
                            'use_mkldnn': False})
        
    def initial_masks(self, program):
        self.mask_value = {}
        percent = 100 * self.ratio
        # percent = [80, 92, 99.1, 93]
        idx = 0
        for param in program.all_parameters():
            if param.name.split('_')[-1] != 'weights':
                continue
            if not (len(param.shape) == 4 and param.shape[2] == 1 
                    and param.shape[3] == 1):
                continue
            weight = np.array(self.scope.find_var(param.name).get_tensor())
            threshold = np.percentile(abs(weight), percent)
            idx += 1
            weight[abs(weight)<threshold] = 0
            mask_value = (weight != 0).astype("float32") # bool -> float32

            self.mask_value[param.name + '_mask'] = mask_value 
            self.scope.var(param.name + '_mask').get_tensor().set(mask_value, self.place)
            print(f"{param.name}_mask:", np.count_nonzero(mask_value)/np.product(param.shape))

    def set_masks(self):
        for name, value in self.mask_value.items():
            self.scope.find_var(name).get_tensor().set(value, self.place)
        print(f"{param.name}_mask:", np.count_nonzero(mask_value)/np.product(value.shape))


    def update_params(self):
        for mask_name in self.mask_value:
            param_name = mask_name.split('_mask')[0]
            param_value = np.array(self.scope.find_var(param_name).get_tensor())
            mask_value = np.array(self.scope.find_var(mask_name).get_tensor())
            param_value = param_value * mask_value
            self.scope.find_var(param_name).get_tensor().set(param_value, self.place)

    @staticmethod
    def total_sparse(program):
        """
        The function is used to get the whole model's sparsity.
        It is static because during testing, we can calculate sparsity without initializing a pruner instance.

        Args:
          - program(paddle.static.Program): The current model.
        Returns:
          - sparsity(float): the model's sparsity.
        """
        total = 0
        values = 0
        for param in program.all_parameters():
            total += np.product(param.shape)
            values += np.count_nonzero(
                np.array(paddle.static.global_scope().find_var(param.name)
                         .get_tensor()))
        sparsity = 1 - float(values) / total
        return sparsity
    
    @staticmethod
    def total_sparse_conv1x1(program):
        """
        The function is used to get the model's spasity for all the 1x1 convolutional weights.
        It is static because during testing, we can calculate sparsity without initializing a pruner instance.

        Args:
          - program(paddle.static.Program): The current model.
        Returns:
          - sparsity(float): the model's sparsity.
        """
        total = 0
        values = 0
        for param in program.all_parameters():
            if not (len(param.shape) == 4 and param.shape[2] == 1 and
                    param.shape[3] == 1):
                continue
            total += np.product(param.shape)
            values += np.count_nonzero(
                np.array(paddle.static.global_scope().find_var(param.name)
                         .get_tensor()))
        sparsity = 1 - float(values) / total
        return sparsity

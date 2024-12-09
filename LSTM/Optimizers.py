import numpy as np

class OptimizerSGDLSTM:
    def __init__(self,learning_rate=1e-5,decay=0,momentum=0) -> None:
        self.learning_rate=learning_rate
        self.current_learning_rate=learning_rate
        self.decay=decay
        self.iterations=0
        self.momentum=momentum
        
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate=self.learning_rate*(1/(1+self.decay*self.iterations))
    def update_params(self,layer):  
            if self.momentum:    
                if not hasattr(layer,"Uf_momentums"): #if there are no momentums to the layer we initalizs them
                    layer.Uf_momentums=np.zeros_like(layer.Uf)
                    layer.Ui_momentums=np.zeros_like(layer.Ui)
                    layer.Uo_momentums=np.zeros_like(layer.Uo)
                    layer.Ug_momentums=np.zeros_like(layer.Ug)


                    layer.Wf_momentums=np.zeros_like(layer.Wf)
                    layer.Wi_momentums=np.zeros_like(layer.Wi)
                    layer.Wo_momentums=np.zeros_like(layer.Wo)
                    layer.Wg_momentums=np.zeros_like(layer.Ug)


                    layer.bf_momentums=np.zeros_like(layer.bf)
                    layer.bi_momentums=np.zeros_like(layer.bi)
                    layer.bo_momentums=np.zeros_like(layer.bo)
                    layer.bg_momentums=np.zeros_like(layer.bg)
                
                #now the momentum parts
                Uf_updates=self.momentum*layer.Uf_momentums-self.current_learning_rate*layer.dUf
                layer.Uf_momentums=Uf_updates

                Ui_updates=self.momentum*layer.Ui_momentums-self.current_learning_rate*layer.dUi
                layer.Ui_momentums=Ui_updates
                
                Uo_updates=self.momentum*layer.Uo_momentums-self.current_learning_rate*layer.dUo
                layer.Uo_momentums=Uo_updates
                
                Ug_updates=self.momentum*layer.Ug_momentums-self.current_learning_rate*layer.dUg
                layer.Ug_momentums=Ug_updates

                Wf_updates=self.momentum*layer.Wf_momentums-self.current_learning_rate*layer.dWf
                layer.Wf_momentums=Wf_updates

                Wi_updates=self.momentum*layer.Wi_momentums-self.current_learning_rate*layer.dWi
                layer.Wi_momentums=Wi_updates
                
                Wo_updates=self.momentum*layer.Wo_momentums-self.current_learning_rate*layer.dWo
                layer.Wo_momentums=Wo_updates
            
                Wg_updates=self.momentum*layer.Wg_momentums-self.current_learning_rate*layer.dWg
                layer.Wg_momentums=Wg_updates

                bf_updates=self.momentum*layer.bf_momentums-self.current_learning_rate*layer.dbf
                layer.bf_momentums=bf_updates

                bi_updates=self.momentum*layer.bi_momentums-self.current_learning_rate*layer.dbi
                layer.bi_momentums=bi_updates
                
                bo_updates=self.momentum*layer.bo_momentums-self.current_learning_rate*layer.dbo
                layer.bo_momentums=bo_updates
                
                bg_updates=self.momentum*layer.bg_momentums-self.current_learning_rate*layer.dbg
                layer.bg_momentums=bg_updates

            else:

                Uf_updates =- self.current_learning_rate*layer.dUf
                Ui_updates =- self.current_learning_rate*layer.dUi
                Uo_updates =- self.current_learning_rate*layer.dUo
                Ug_updates =- self.current_learning_rate*layer.dUg
                Wf_updates =- self.current_learning_rate*layer.dWf
                Wi_updates =- self.current_learning_rate*layer.dWi
                Wo_updates =- self.current_learning_rate*layer.dWo
                Wg_updates =- self.current_learning_rate*layer.dWg    
                bf_updates =- self.current_learning_rate*layer.dbf
                bi_updates =- self.current_learning_rate*layer.dbi
                bo_updates =- self.current_learning_rate*layer.dbo
                bg_updates =- self.current_learning_rate*layer.dbg

               
            layer.dUf += Uf_updates
            layer.dUi += Ui_updates
            layer.dUo += Uo_updates
            layer.dUg += Ug_updates
            layer.dWf += Wf_updates
            layer.dWi += Wi_updates
            layer.dWo += Wo_updates
            layer.dWg += Wg_updates   
            layer.dbf += bf_updates
            layer.dbi += bi_updates
            layer.dbo += bo_updates
            layer.dbg += bg_updates

    def post_update_params(self):
        self.iterations+=1

class OptimizerSGD:

        def __init__(self,learning_rate=1e-5,decay=0,momentum=0) -> None:
            self.learning_rate=learning_rate
            self.current_learning_rate=learning_rate
            self.decay=decay
            self.iterations=0
            self.momentum=momentum

        def pre_update_params(self):
            if self.decay:
              self.current_learning_rate=self.learning_rate*(1/(1+self.decay*self.iterations))

        def update_params(self,layer):

            if self.momentum:
                if not hasattr(layer,'weight_momentums'):
                  layer.weight_momentums=np.zeros_like(layer.weights)
                  layer.bais_momentums=np.zeros_like(layer.biases)
                weight_updates=self.momentum*layer.weight_momentums-self.current_learning_rate*layer.dweights
                layer.weight_momentums=weight_updates

                bias_updates=self.momentum*layer.bais_momentums-self.current_learning_rate*layer.dbiases
                layer.bias_momentums=bias_updates

            else:
                weight_updates=-self.current_learning_rate*layer.dweights
                bias_updates=-self.current_learning_rate*layer.dbiases
            layer.weights+=weight_updates
            layer.biases+=bias_updates
        
        def post_update_params(self):
            self.iterations+=1
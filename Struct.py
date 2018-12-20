#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 16:33:18 2018

@author: liyuan
"""
import numpy as np

class Struct:
    def __str__(self):
        """
        print attributes
        """
        attr_list = []
        for attribute in dir(self):
            if attribute != '__weakref__':
                if not hasattr(getattr(self,attribute),'__call__') and not '__' in attribute :
                    attribute_value = getattr(self,attribute)
                    if isinstance(attribute_value, (np.ndarray, np.generic)):
                        attribute_value = 'array with shape {}'.format(str(attribute_value.shape))
                    elif issubclass(type(attribute_value), Struct):
                        attribute_value = 'Struct with {} attributes'.format(len(attribute_value))
                    elif isinstance(attribute_value, list):
                        attribute_value = 'list with {} elements'.format(len(attribute_value))

                    line = '{:>35s} : {:<10s}'.format(attribute,str(attribute_value))
                    attr_list.append(line)
        return '\n'.join(attr_list)
    
    def __len__(self):
        count = 0
        for attribute in dir(self):
            if not hasattr(getattr(self,attribute),'__call__') and not '__' in attribute:
                count = count + 1
        return count
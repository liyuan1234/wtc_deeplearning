#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 16:33:18 2018

@author: liyuan
"""
class Struct:
    def __str__(self):
        attr_list = []
        for attribute in dir(self):
            if not hasattr(getattr(self,attribute),'__call__') and not '__' in attribute:
                line = '{:>35s} : {:<10s}'.format(attribute,str(getattr(self,attribute)))
                attr_list.append(line)
        if attr_list:
            return '\n'.join(attr_list)+'\n'
        else:
            return ''
    
    def __len__(self):
        count = 0
        for attribute in dir(self):
            if not hasattr(getattr(self,attribute),'__call__') and not '__' in attribute:
                count = count + 1
        return count
package com.example.ro.dto.responseDTO;

import com.example.ro.enumeration.Role;
import lombok.Data;

@Data
public class LinkOfTreeDTO {

    private int id;
    private int id_source;
    private int id_target;
    private Role RelationType;
    private int weight;
}

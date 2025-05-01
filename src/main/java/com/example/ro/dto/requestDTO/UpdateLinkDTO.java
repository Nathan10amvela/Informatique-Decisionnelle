package com.example.ro.dto.requestDTO;

import com.example.ro.enumeration.Role;

import lombok.Data;

@Data
public class UpdateLinkDTO {


    private Role relationType;

    private PersonDTO source;

    private PersonDTO target;
}

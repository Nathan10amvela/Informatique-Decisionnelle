package com.example.ro.dto.requestDTO;

import com.example.ro.enumeration.Role;
import lombok.Data;

import jakarta.validation.constraints.NotNull;
import jakarta.validation.constraints.Positive;
import jakarta.validation.constraints.Size;

@Data
public class LinkDTO {


    @NotNull(message = "Relation type cannot be null.")
    private Role relationType;


    private int id_source;

    private PersonDTO source;

    private int id_target;

    private PersonDTO target;

    @Positive(message = "Family tree ID must be a positive number.")
    @NotNull(message = "Family Tree ID cannot be null.")
    private int familyTreeId;

}

package com.example.ro.dto.requestDTO;

import com.example.ro.enumeration.Gender;
import com.example.ro.enumeration.Role;

import lombok.Data;

@Data
public class PersonDTO {
    private String lastName;

    private String firstName;

    //private Instant birthDate;

    private String birthPlace;

    private Gender gender;

    private Role role;
}

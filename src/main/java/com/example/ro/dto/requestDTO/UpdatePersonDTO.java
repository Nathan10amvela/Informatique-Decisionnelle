package com.example.ro.dto.requestDTO;

import com.example.ro.enumeration.Gender;
import com.example.ro.enumeration.Role;
import lombok.Data;

import java.sql.Date;

@Data
public class UpdatePersonDTO {

    private String lastName;

    private String firstName;

    private String birthDate;

    private String birthPlace;

    private Gender gender;

}

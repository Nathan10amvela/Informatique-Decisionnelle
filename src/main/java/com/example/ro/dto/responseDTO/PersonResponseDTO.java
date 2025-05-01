package com.example.ro.dto.responseDTO;

import com.example.ro.enumeration.Gender;
import lombok.Data;

@Data
public class PersonResponseDTO {

    private int id;
    private String lastName;
    private String firstName;
    private String birthDate;
    private String birthPlace;
    private Gender gender;
}

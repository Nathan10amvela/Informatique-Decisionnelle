package com.example.ro.dto.requestDTO;

import com.example.ro.enumeration.Gender;
import com.example.ro.enumeration.Role;

import jakarta.validation.constraints.NotNull;
import jakarta.validation.constraints.Past;
import jakarta.validation.constraints.Positive;
import jakarta.validation.constraints.Size;
import lombok.Data;

import java.sql.Date;

@Data
public class PersonDTO {

    @NotNull(message = "Last name cannot be null.")
    @Size(min = 1, max = 100, message = "Last name must be between 1 and 100 characters.")
    private String lastName;

    @NotNull(message = "First name cannot be null.")
    @Size(min = 1, max = 100, message = "First name must be between 1 and 100 characters.")
    private String firstName;

    @NotNull(message = "Birth date cannot be null.")
    @Past(message = "Birth date must be in the past.")
    private String birthDate;

    @NotNull(message = "Birth place cannot be null.")
    @Size(min = 1, max = 150, message = "Birth place must be between 1 and 150 characters.")
    private String birthPlace;

    @NotNull(message = "Gender cannot be null.")
    private Gender gender;

}
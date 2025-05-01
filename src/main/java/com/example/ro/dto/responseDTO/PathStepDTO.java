package com.example.ro.dto.responseDTO;

import com.example.ro.enumeration.Role;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@AllArgsConstructor
@NoArgsConstructor
public class PathStepDTO {
    private String fromPerson;
    private String toPerson;
    private Role relationType;
    private int weight;
}
